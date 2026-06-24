"""
GW-VLM 训练脚本：Unsloth FastVisionModel + LoRA 视觉微调。
对 Qwen3-VL / Qwen3.6 / Gemma 4 通用——同一份 08 导出的 messages jsonl 即可喂。

数据：08_export_training_format.py 产出的 output/training_data/{e1,e2}/{train,val,test}.jsonl
  每行 {"messages":[system, user(content=[image,text?]), assistant]}，
  其中 image 为相对路径（相对 --image-root，默认相对 spectrograms/）。

用法（在 DGX Spark 上）：
  python train_vlm.py --config configs/e1_gemma4_e4b.yaml \
      --image-root ~/GW-VLM/output/spectrograms

  # 冒烟测试（只取少量样本、1 epoch）：
  python train_vlm.py --config configs/e1_gemma4_e4b.yaml \
      --image-root ~/GW-VLM/output/spectrograms --max-samples 32 --epochs 1

任何 YAML 配置项都可被同名 CLI 覆盖（见 build_arg_parser）。
"""
import argparse
import json
import os
from pathlib import Path

import yaml

# 关闭 Unsloth 匿名统计上报：受限网络下它会 snapshot_download 遥测仓库、卡 120s 后
# 误报 "HuggingFace seems to be down"。必须在 import unsloth 前设置。
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
# 关闭 HF Xet 下载后端：其 rust 客户端不走 HTTP(S)_PROXY，在代理网络下会 0 字节卡死。
# 改用经典 HTTPS resolve（遵守代理）。如在 Xet 直连可用的网络可去掉本行。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# DGX Spark/GB10 统一内存：大 bf16 模型的 device_map 会把权重"offload 到 CPU"(其实同一块
# 物理内存,无害),accelerate 会误判分布式+多设备而拒训。设此变量跳过该检查(官方逃生门)。
# 写在脚本里而非仅靠 shell export，确保一定进入进程环境（accelerate 在 prepare 时读取）。
os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")


# ---- 配置加载 / CLI 覆盖 ----------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unsloth VLM LoRA 训练（GW-VLM）")
    ap.add_argument("--config", type=Path, required=True, help="YAML 配置路径")
    # 可覆盖项（None=用配置里的值）
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--data-dir", default=None, help="含 train.jsonl/val.jsonl 的目录")
    ap.add_argument("--image-root", default=None, help="相对 image 路径的基准目录")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="只取前 N 条训练样本（冒烟测试）")
    ap.add_argument("--report-to", default=None, help="none / wandb")
    ap.add_argument("--no-eval", action="store_true", help="禁用验证集评估")
    return ap


def merge_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # CLI 覆盖（仅当显式提供）
    overrides = {
        "model_id": args.model_id,
        "data_dir": args.data_dir,
        "image_root": args.image_root,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "report_to": args.report_to,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.no_eval:
        cfg["do_eval"] = False
    cfg.setdefault("seed", 42)
    cfg.setdefault("lora_dropout", 0.0)
    cfg.setdefault("max_seq_length", 2048)
    cfg.setdefault("do_eval", True)
    cfg.setdefault("report_to", "none")
    return cfg, args.max_samples


# ---- 数据：jsonl → 含 PIL 图像的会话列表 -----------------------------------

def load_conversations(jsonl_path: Path, image_root: str, max_samples=None) -> list:
    """读 messages jsonl，把 user content 里的 image 相对路径替换成 PIL.Image。"""
    from PIL import Image

    convs = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rec = json.loads(line)
            for msg in rec["messages"]:
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image":
                            p = part["image"]
                            if not os.path.isabs(p):
                                p = os.path.join(image_root, p)
                            part["image"] = Image.open(p).convert("RGB")
            convs.append({"messages": rec["messages"]})
    return convs


# ---- 主流程 ----------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    cfg, max_samples = merge_config(args)

    if "image_root" not in cfg:
        raise SystemExit("必须提供 --image-root 或在配置里写 image_root")
    data_dir = Path(cfg["data_dir"])
    output_dir = cfg["output_dir"]

    print(f"[train_vlm] model={cfg['model_id']}  data={data_dir}  image_root={cfg['image_root']}")
    print(f"[train_vlm] lora r={cfg['lora_r']} alpha={cfg['lora_alpha']} dropout={cfg['lora_dropout']}  "
          f"4bit={cfg.get('load_in_4bit', False)}  seq_len={cfg['max_seq_length']}")

    # 延迟导入：仅在真正训练时需要 GPU 栈（本地无 GPU 也能 import 本模块做语法检查）
    import torch
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    fp_kwargs = dict(
        model_name=cfg["model_id"],
        load_in_4bit=cfg.get("load_in_4bit", False),
        use_gradient_checkpointing="unsloth",
        max_seq_length=cfg["max_seq_length"],
    )
    # 大 bf16 模型在 GB10 统一内存上，默认 device_map="sequential" 会误判显存把权重
    # offload 到 CPU → 多设备切分 → accelerate 拒训。配 device_map: {"": 0} 强制全放 GPU。
    if cfg.get("device_map") is not None:
        fp_kwargs["device_map"] = cfg["device_map"]
    model, processor = FastVisionModel.from_pretrained(**fp_kwargs)

    # Gemma 4 等需要指定 chat 模板时设 cfg["chat_template"]；Qwen 系列 Unsloth 自动处理
    if cfg.get("chat_template"):
        from unsloth.chat_templates import get_chat_template
        processor = get_chat_template(processor, cfg["chat_template"])

    # 输入分辨率：Gemma4 按 token 预算(70/140/280/560/1120)决定有效分辨率
    # (560≈1135px 正好吃满 1024 源)。设了就改处理器预算 + 让 collator resize="max"
    # 关掉 Unsloth 默认对 Gemma4 的 512 兜底(偏低)。
    img_tokens = cfg.get("image_tokens")
    collator_kwargs = {}
    if img_tokens is not None:
        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "max_soft_tokens"):
            ip.max_soft_tokens = img_tokens
            ip.image_seq_length = img_tokens
        if hasattr(processor, "image_seq_length"):
            processor.image_seq_length = img_tokens
        collator_kwargs["resize"] = "max"
        print(f"[train_vlm] image_tokens={img_tokens}（resize=max，关闭 Unsloth 512 兜底）")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.get("finetune_vision_layers", True),
        finetune_language_layers=cfg.get("finetune_language_layers", True),
        finetune_attention_modules=cfg.get("finetune_attention_modules", True),
        finetune_mlp_modules=cfg.get("finetune_mlp_modules", True),
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        random_state=cfg["seed"],
        target_modules=cfg.get("target_modules", "all-linear"),
    )

    train_convs = load_conversations(data_dir / "train.jsonl", cfg["image_root"], max_samples)
    eval_convs = None
    if cfg["do_eval"]:
        eval_convs = load_conversations(data_dir / "val.jsonl", cfg["image_root"])
    print(f"[train_vlm] train={len(train_convs)}  eval={len(eval_convs) if eval_convs else 0}")

    FastVisionModel.for_training(model)
    sft_args = SFTConfig(
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        num_train_epochs=cfg["epochs"],
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=cfg.get("logging_steps", 5),
        optim=cfg.get("optim", "adamw_8bit"),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        seed=cfg["seed"],
        output_dir=output_dir,
        report_to=cfg["report_to"],
        bf16=cfg.get("bf16", not cfg.get("load_in_4bit", False)),
        # 视觉 SFT 必需
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_length=cfg["max_seq_length"],
        save_strategy="epoch",
        eval_strategy="epoch" if cfg["do_eval"] else "no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor, **collator_kwargs),
        train_dataset=train_convs,
        eval_dataset=eval_convs,
        args=sft_args,
    )

    stats = trainer.train()
    print(f"[train_vlm] done. train_runtime={stats.metrics.get('train_runtime')}s  "
          f"loss={stats.metrics.get('train_loss')}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[train_vlm] LoRA adapter 已保存到 {output_dir}")


if __name__ == "__main__":
    main()
