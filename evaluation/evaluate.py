"""
E1 检测评估：加载 base+LoRA adapter，对 test 集逐条推理，解析 detection，出指标。

用法（DGX Spark）：
  python evaluation/evaluate.py --adapter output/runs/e1_gemma4_e4b \
      --test output/training_data/e1/test.jsonl \
      --image-root ~/GW-VLM/output/spectrograms

输出：<out-dir>/eval_report.json + confusion.png（默认 out-dir = adapter 目录）。
本轮先做 clean test；MLGWSC/Glitch/OOD 鲁棒性场景留后续。
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")  # 受限网络下避免遥测 120s 超时误报
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 代理网络下 Xet 会卡死，改走经典 HTTPS

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 `from metrics import ...` 生效
from metrics import parse_detection, compute_metrics, save_confusion_png


def load_test(path: Path, image_root: str, max_samples=None) -> list:
    """读 messages jsonl，拆成 (prompt_messages 含 PIL 图, gold detection)。"""
    from PIL import Image

    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rec = json.loads(line)
            gold, prompt_msgs = None, []
            for m in rec["messages"]:
                if m["role"] == "assistant":
                    try:
                        gold = json.loads(m["content"]).get("detection")
                    except Exception:
                        gold = None
                    continue
                content = m["content"]
                if isinstance(content, list):
                    new_content = []
                    for part in content:
                        if part.get("type") == "image":
                            p = part["image"]
                            if not os.path.isabs(p):
                                p = os.path.join(image_root, p)
                            new_content.append({"type": "image", "image": Image.open(p).convert("RGB")})
                        else:
                            new_content.append(part)
                    prompt_msgs.append({"role": m["role"], "content": new_content})
                else:
                    prompt_msgs.append(m)
            items.append({"messages": prompt_msgs, "gold": gold})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="GW-VLM E1 检测评估")
    ap.add_argument("--adapter", required=True, help="LoRA adapter 目录（或直接给 base model id 做 zero-shot）")
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true",
                    help="以 bf16 加载基座（默认 4bit，与多数 E4B/调试 adapter 训练一致）")
    ap.add_argument("--image-tokens", type=int, default=None,
                    help="Gemma4 图像 token 预算(70/140/280/560/1120)；须与训练一致")
    args = ap.parse_args()

    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(args.adapter, load_in_4bit=not args.no_4bit)
    FastVisionModel.for_inference(model)

    # 输入分辨率须与训练一致（Gemma4 token 预算）
    if args.image_tokens is not None:
        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "max_soft_tokens"):
            ip.max_soft_tokens = args.image_tokens
            ip.image_seq_length = args.image_tokens
        if hasattr(processor, "image_seq_length"):
            processor.image_seq_length = args.image_tokens
        print(f"[evaluate] image_tokens={args.image_tokens}")

    items = load_test(args.test, args.image_root, args.max_samples)
    y_true, y_pred, n_unparsed = [], [], 0
    for k, it in enumerate(items):
        inputs = processor.apply_chat_template(
            it["messages"], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        pred = parse_detection(text)
        if pred is None:
            n_unparsed += 1
            pred = "NO"  # 无法解析按 NO 兜底
        y_true.append(it["gold"])
        y_pred.append(pred)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(items)} ...")

    metrics = compute_metrics(y_true, y_pred)
    metrics["n_unparsed"] = n_unparsed

    out_dir = Path(args.out_dir or args.adapter)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    save_confusion_png(metrics["confusion_matrix"]["matrix"], out_dir / "confusion.png")

    print("\n=== E1 检测指标 ===")
    for key in ("n", "accuracy", "precision_YES", "recall_YES", "f1_YES"):
        print(f"  {key}: {metrics[key]}")
    print(f"  confusion (行真实/列预测 NO,YES): {metrics['confusion_matrix']['matrix']}")
    print(f"  无法解析(按NO兜底): {n_unparsed}")
    print(f"写入 {out_dir}/eval_report.json + confusion.png")


if __name__ == "__main__":
    main()
