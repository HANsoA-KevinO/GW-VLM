"""
多模态融合(方法2)训练:Qwen2.5-VL + 1D 应变编码器,原生 transformers + PEFT。
消融:use_image / use_strain 任意组合(C=仅图 / A=仅应变 / B=图+应变)。

用法(Spark):
  python training/train_fusion.py --config training/configs/fusion_qwen2.5vl_3b.yaml \
      --use-image true --use-strain true --output-dir output/runs/fusion_B --epochs 3
冒烟:加 --max-samples 32 --epochs 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def b(x):
    return str(x).lower() in ("1", "true", "yes", "y")


def build_dataset(jsonl, image_root, strain_root, use_image, use_strain, max_samples=None):
    from PIL import Image
    rows = []
    for i, line in enumerate(open(jsonl)):
        if max_samples is not None and i >= max_samples:
            break
        rec = json.loads(line)
        msgs = rec["messages"]
        img_rel = None
        for m in msgs:
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "image":
                        img_rel = part["image"]
        # 重建 messages(按消融决定是否带图)
        new_msgs = []
        for m in msgs:
            c = m["content"]
            if isinstance(c, list):
                if use_image:
                    nc = []
                    for part in c:
                        if part.get("type") == "image":
                            p = part["image"]
                            p = p if os.path.isabs(p) else os.path.join(image_root, p)
                            nc.append({"type": "image", "image": Image.open(p).convert("RGB")})
                        else:
                            nc.append(part)
                    new_msgs.append({"role": m["role"], "content": nc})
                else:
                    new_msgs.append({"role": m["role"],
                                     "content": [{"type": "text", "text": "Analyze the gravitational-wave data."}]})
            else:
                new_msgs.append(m)
        ex = {"messages": new_msgs,
              "images": ([im["image"] for mm in new_msgs if isinstance(mm["content"], list)
                          for im in mm["content"] if im.get("type") == "image"] if use_image else None),
              "use_strain": use_strain}
        if use_strain and img_rel:
            arr = __import__("numpy").load(os.path.join(strain_root, Path(img_rel).stem + ".npy"))
            ex["strain"] = arr
        rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--use-image", default=None)
    ap.add_argument("--use-strain", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--strain-patch", type=int, default=None)   # 覆盖 patch_size(32:256 / 64:128)
    ap.add_argument("--load-4bit", default=None)                # 覆盖 load_in_4bit
    ap.add_argument("--grad-ckpt", default=None)                # 覆盖 gradient_checkpointing
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.strain_patch is not None:
        cfg["strain_patch_size"] = args.strain_patch
    if args.load_4bit is not None:
        cfg["load_in_4bit"] = b(args.load_4bit)
    if args.grad_ckpt is not None:
        cfg["gradient_checkpointing"] = b(args.grad_ckpt)
    use_image = b(args.use_image) if args.use_image is not None else cfg["use_image"]
    use_strain = b(args.use_strain) if args.use_strain is not None else cfg["use_strain"]
    out_dir = Path(args.output_dir or cfg["output_dir"])
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fusion] use_image={use_image} use_strain={use_strain} out={out_dir} epochs={epochs}", flush=True)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from models.strain_encoder import StrainEncoder1D, StrainPatchEncoder
    from fusion_model import FusionVLM, repair_misquantized_linears
    from fusion_collator import FusionCollator

    family = cfg.get("model_family", "qwen")
    load_4bit = b(cfg.get("load_in_4bit", False))
    quant = None
    if load_4bit:   # 27B QLoRA:权重 ~55GB→~14GB,留足余量、消除 OOM、缓解统一内存压力(也救了 sshd)
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"])  # 视觉塔保 bf16,图像质量不受量化影响
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant)
    print(f"[fusion] load_in_4bit={load_4bit}", flush=True)
    if family == "gemma":
        nfix = repair_misquantized_linears(model)
        print(f"[fusion] 修复误包 4bit 层(vision/audio 塔): {nfix}", flush=True)
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    # Qwen:限制图像 token 数提速(消融内部一致)。Gemma:无 max_pixels,走原生全分辨率。
    ip = getattr(processor, "image_processor", None)
    if family == "qwen" and ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = int(cfg.get("max_pixels", 262144))   # ~83 个图 token
    model.config.use_cache = False   # 不开梯度检查点:3B 显存充裕,省去前向重算更快

    if family == "gemma":
        # Gemma4 的 audio_tower 用 Gemma4ClippableLinear(PEFT 不支持);用正则把 LoRA
        # 限定在 language_model 的 q/k/v/o + gate/up/down(LLM 内的是普通 Linear4bit/Linear)。
        target_modules = (r".*language_model\.layers\.\d+\."
                          r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if b(cfg.get("gradient_checkpointing", False)):
        # bf16 27B:省激活,把 forward 压到加载峰值以下(use_reentrant=False 兼容 inputs_embeds)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()   # FusionVLM 内会调 get_input_embeddings → 让其输出 requires_grad
        print("[fusion] gradient_checkpointing=ON", flush=True)

    hidden = model.get_base_model().config.text_config.hidden_size
    img_tok = model.get_base_model().config.image_token_id

    # 应变编码器:按 strain_encoder_type 选(patch_attn=第2轮 patch+自注意力;cnn=第1轮)
    enc_type = cfg.get("strain_encoder_type", "cnn")
    in_len = cfg["strain_in_len"]
    if enc_type == "patch_attn":
        patch_size = int(cfg["strain_patch_size"])
        n_tokens = in_len // patch_size          # 自动推导(256→32, 128→64)
    else:
        n_tokens = cfg["strain_n_tokens"]
    strain_enc = None
    if use_strain:
        if enc_type == "patch_attn":
            strain_enc = StrainPatchEncoder(
                hidden, in_len=in_len, patch_size=patch_size,
                n_attn_layers=int(cfg.get("strain_attn_layers", 3)),
                n_heads=int(cfg.get("strain_attn_heads", 8)),
                mlp_proj=b(cfg.get("strain_mlp_proj", True))).to("cuda", torch.float32)
        else:
            strain_enc = StrainEncoder1D(hidden, n_tokens=n_tokens, in_len=in_len,
                                         channels=tuple(cfg["strain_channels"])).to("cuda", torch.float32)
    fusion = FusionVLM(model, strain_enc, img_tok, model_family=family)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collator = FusionCollator(processor, n_tokens, img_tok, pad_id,
                              enable_thinking=cfg.get("enable_thinking", False), family=family)
    ds = build_dataset(Path(cfg["data_dir"]) / "train.jsonl", cfg["image_root"], cfg["strain_root"],
                       use_image, use_strain, args.max_samples)
    print(f"[fusion] 训练样本 {len(ds)}  hidden={hidden} img_tok={img_tok}", flush=True)
    nw = int(cfg.get("num_workers", 4))   # 27B 设 0:前向才是瓶颈,且避免崩溃时留下孤儿 worker 拖垮机器
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collator,
                    num_workers=nw, persistent_workers=(nw > 0))

    # 两组参数:LoRA / StrainEncoder
    lora_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": float(cfg["learning_rate"])}]
    if strain_enc is not None:
        groups.append({"params": list(strain_enc.parameters()), "lr": float(cfg["strain_encoder_lr"])})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    ga = cfg["grad_accum"]
    total_steps = int(len(dl) * epochs / ga)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=max(1, total_steps),
        pct_start=cfg.get("warmup_ratio", 0.03))

    fusion.train()
    step = 0
    for ep in range(int(epochs)):
        for it, batch in enumerate(dl):
            batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
            out = fusion(**batch)
            loss = out.loss / ga
            loss.backward()
            if (it + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_params + (list(strain_enc.parameters()) if strain_enc else []), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"  ep{ep} step{step} loss={out.loss.item():.4f}", flush=True)
    # 保存
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    if strain_enc is not None:
        torch.save(strain_enc.state_dict(), out_dir / "strain_encoder.pt")
    json.dump({"use_image": use_image, "use_strain": use_strain,
               "model_family": family, "model_id": cfg["model_id"], "load_in_4bit": load_4bit,
               "strain_encoder_type": enc_type, "strain_n_tokens": n_tokens,
               "strain_in_len": in_len,
               "strain_patch_size": int(cfg.get("strain_patch_size", 0)),
               "strain_attn_layers": int(cfg.get("strain_attn_layers", 3)),
               "strain_attn_heads": int(cfg.get("strain_attn_heads", 8)),
               "strain_mlp_proj": b(cfg.get("strain_mlp_proj", True)),
               "strain_channels": cfg.get("strain_channels"),
               "max_pixels": int(cfg.get("max_pixels", 262144))},
              open(out_dir / "fusion_meta.json", "w"), indent=2)
    print(f"[fusion] done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
