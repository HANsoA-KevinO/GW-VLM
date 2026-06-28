"""
统一训练(E5):Qwen3.6-27B 融合(图 ⊕ 应变)+ **后验头**(VLA 式参数估计)。
- 检测:维持生成式 {"detection":"YES/NO"}(LM 交叉熵)。
- 参数:从最后一个 prompt token 的末层 hidden 接 GaussianPosteriorHead,出 (μ,logσ)×3,
  高斯 NLL(标准化空间)只在正样本上算;总损失 = LM-CE + λ·NLL。
fork 自 train_fusion.py;复用 FusionVLM/FusionCollator/StrainEncoder。

用法(Spark):
  python training/train_unified.py --config training/configs/unified_qwen36_27b.yaml \
      --output-dir output/runs/unified_q27 --epochs 2
冒烟:--max-samples 32 --epochs 1 --load-4bit true
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


def compute_norm_stats(jsonl):
    """扫 train jsonl 的正样本 targets,算标准化统计(变换空间 mean/std)。"""
    import numpy as np
    from models.posterior_head import compute_norm_stats as _cns, PARAM_NAMES
    vals = []
    for line in open(jsonl):
        t = json.loads(line).get("targets")
        if t and all(t.get(n) is not None for n in PARAM_NAMES):
            vals.append([float(t[n]) for n in PARAM_NAMES])
    if not vals:
        raise RuntimeError(f"{jsonl} 无正样本 targets,无法算 norm_stats")
    return _cns(np.asarray(vals, dtype=float))


def build_dataset(jsonl, image_root, strain_root, use_image, use_strain, stats, max_samples=None):
    import numpy as np
    from PIL import Image
    from models.posterior_head import PARAM_NAMES, standardize
    rows = []
    for i, line in enumerate(open(jsonl)):
        if max_samples is not None and i >= max_samples:
            break
        rec = json.loads(line)
        msgs = rec["messages"]
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
        base = rec.get("basename")
        if use_strain and base:
            ex["strain"] = np.load(os.path.join(strain_root, base + ".npy"))
        # 后验头目标:标准化 z(NaN 表示负样本无参数)
        t = rec.get("targets") or {}
        is_pos = all(t.get(n) is not None for n in PARAM_NAMES)
        phys = np.array([[float(t[n]) if is_pos else np.nan for n in PARAM_NAMES]], dtype=float)
        ex["param_target"] = standardize(phys, stats)[0].astype("float32")
        ex["param_valid"] = bool(is_pos)
        rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--strain-patch", type=int, default=None)
    ap.add_argument("--load-4bit", default=None)
    ap.add_argument("--grad-ckpt", default=None)
    ap.add_argument("--param-loss-weight", type=float, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.strain_patch is not None:
        cfg["strain_patch_size"] = args.strain_patch
    if args.load_4bit is not None:
        cfg["load_in_4bit"] = b(args.load_4bit)
    if args.grad_ckpt is not None:
        cfg["gradient_checkpointing"] = b(args.grad_ckpt)
    if args.param_loss_weight is not None:
        cfg["param_loss_weight"] = args.param_loss_weight
    use_image = b(cfg.get("use_image", True))
    use_strain = b(cfg.get("use_strain", True))
    out_dir = Path(args.output_dir or cfg["output_dir"])
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    plw = float(cfg.get("param_loss_weight", 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unified] use_image={use_image} use_strain={use_strain} out={out_dir} epochs={epochs} λ={plw}", flush=True)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from models.strain_encoder import StrainEncoder1D, StrainPatchEncoder
    from models.posterior_head import GaussianPosteriorHead, PARAM_NAMES
    from fusion_model import FusionVLM, repair_misquantized_linears
    from fusion_collator import FusionCollator

    family = cfg.get("model_family", "qwen")
    load_4bit = b(cfg.get("load_in_4bit", False))
    quant = None
    if load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant)
    print(f"[unified] load_in_4bit={load_4bit}", flush=True)
    if family == "gemma":
        repair_misquantized_linears(model)
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    ip = getattr(processor, "image_processor", None)
    if family == "qwen" and ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = int(cfg.get("max_pixels", 262144))
    model.config.use_cache = False

    if family == "gemma":
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
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("[unified] gradient_checkpointing=ON", flush=True)

    hidden = model.get_base_model().config.text_config.hidden_size
    img_tok = model.get_base_model().config.image_token_id

    enc_type = cfg.get("strain_encoder_type", "patch_attn")
    in_len = cfg["strain_in_len"]
    if enc_type == "patch_attn":
        patch_size = int(cfg["strain_patch_size"])
        n_tokens = in_len // patch_size
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

    # 后验头
    head = GaussianPosteriorHead(hidden, n_params=len(PARAM_NAMES),
                                 mlp_hidden=int(cfg.get("head_mlp_hidden", 256)),
                                 dropout=float(cfg.get("head_dropout", 0.1))).to("cuda", torch.float32)
    fusion = FusionVLM(model, strain_enc, img_tok, model_family=family,
                       param_head=head, param_loss_weight=plw)

    # norm_stats(用全量 train 正样本算,稳)
    train_jsonl = Path(cfg["data_dir"]) / "train.jsonl"
    stats = compute_norm_stats(train_jsonl)
    json.dump(stats, open(out_dir / "norm_stats.json", "w"), indent=2)
    print(f"[unified] norm_stats(变换空间) mean={['%.3f'%m for m in stats['mean']]} "
          f"std={['%.3f'%s for s in stats['std']]}", flush=True)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collator = FusionCollator(processor, n_tokens, img_tok, pad_id,
                              enable_thinking=cfg.get("enable_thinking", False), family=family)
    ds = build_dataset(train_jsonl, cfg["image_root"], cfg["strain_root"],
                       use_image, use_strain, stats, args.max_samples)
    npos = sum(1 for e in ds if e["param_valid"])
    print(f"[unified] 训练样本 {len(ds)}(正 {npos})  hidden={hidden} img_tok={img_tok}", flush=True)
    nw = int(cfg.get("num_workers", 0))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collator,
                    num_workers=nw, persistent_workers=(nw > 0))

    # 三组参数:LoRA / StrainEncoder / 后验头
    lora_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": float(cfg["learning_rate"])}]
    if strain_enc is not None:
        groups.append({"params": list(strain_enc.parameters()), "lr": float(cfg["strain_encoder_lr"])})
    groups.append({"params": list(head.parameters()), "lr": float(cfg.get("head_lr", 1.0e-3))})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    ga = cfg["grad_accum"]
    total_steps = max(1, int(len(dl) * epochs / ga))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=total_steps,
        pct_start=cfg.get("warmup_ratio", 0.03))
    clip_params = lora_params + (list(strain_enc.parameters()) if strain_enc else []) + list(head.parameters())

    fusion.train()
    step = 0
    for ep in range(int(epochs)):
        for it, batch in enumerate(dl):
            batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
            out = fusion(**batch)
            loss = out.loss / ga
            loss.backward()
            if (it + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    nll = getattr(out, "param_nll", None)
                    nll_s = f" nll={nll.item():.4f}" if nll is not None else ""
                    print(f"  ep{ep} step{step} loss={out.loss.item():.4f}{nll_s}", flush=True)

    # 保存
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    if strain_enc is not None:
        torch.save(strain_enc.state_dict(), out_dir / "strain_encoder.pt")
    torch.save(head.state_dict(), out_dir / "posterior_head.pt")
    json.dump({"use_image": use_image, "use_strain": use_strain,
               "model_family": family, "model_id": cfg["model_id"], "load_in_4bit": load_4bit,
               "has_param_head": True, "param_names": PARAM_NAMES, "param_loss_weight": plw,
               "head_mlp_hidden": int(cfg.get("head_mlp_hidden", 256)),
               "strain_encoder_type": enc_type, "strain_n_tokens": n_tokens, "strain_in_len": in_len,
               "strain_patch_size": int(cfg.get("strain_patch_size", 0)),
               "strain_attn_layers": int(cfg.get("strain_attn_layers", 3)),
               "strain_attn_heads": int(cfg.get("strain_attn_heads", 8)),
               "strain_mlp_proj": b(cfg.get("strain_mlp_proj", True)),
               "strain_channels": cfg.get("strain_channels"),
               "max_pixels": int(cfg.get("max_pixels", 262144))},
              open(out_dir / "fusion_meta.json", "w"), indent=2)
    print(f"[unified] done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
