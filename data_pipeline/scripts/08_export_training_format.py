"""
把中间格式 dataset_{train,val,test}.jsonl 转成**训练器可直接吃的会话格式**。

输出统一的 messages 结构（Qwen3-VL / Qwen3.6 与 Gemma 4 通用——两者都接受
content=[{type:image},{type:text}]，Gemma 强制"图像在前"，Qwen 兼容）：

  {
    "messages": [
      {"role": "system",    "content": "<固定 system prompt>"},
      {"role": "user",      "content": [{"type": "image", "image": "<相对/绝对路径>.png"}]},
      {"role": "assistant", "content": "{\"detection\": \"YES\"}"}
    ]
  }

schema：
  - detection_only (E1/E3)：assistant = {"detection": "YES"|"NO"}
  - multitask     (E2/E4)：assistant = {"detection", "chirp_mass_bin", "distance_bin", "chi_eff_bin"}

system prompt 取自 docs/02_research_design.md §4.1（E1 用精简版只问 detection；E2 用完整版）。

image 路径：
  --image-path relative（默认）→ 写成相对 --image-base（默认 spectrograms/）的路径，
    训练脚本用 --image-root 拼回绝对路径，规避跨机器（Mac→DGX Spark）路径迁移问题。
  --image-path absolute → 保留 dataset.jsonl 里的绝对路径（本机直接训练用）。

用法：
  python 08_export_training_format.py --schema detection_only          # → output/training_data/e1/
  python 08_export_training_format.py --schema multitask               # → output/training_data/e2/
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_DIR, SPECTROGRAMS_DIR


SPLITS = ("train", "val", "test")
SCHEMA_TO_SUBDIR = {"detection_only": "e1", "multitask": "e2"}

# --- 固定 system prompt（docs/02 §4.1）-------------------------------------

SYSTEM_PROMPT_DETECTION_ONLY = (
    "You are an expert gravitational-wave data analyst. Given a Q-transform "
    "time-frequency spectrogram from LIGO/Virgo strain data, determine whether "
    "it contains a gravitational wave signal.\n\n"
    "Output strictly as a JSON object with a single field:\n"
    '- "detection": "YES" or "NO"\n\n'
    "Output only the JSON, no other text."
)

SYSTEM_PROMPT_MULTITASK = (
    "You are an expert gravitational-wave data analyst. Given a Q-transform "
    "time-frequency spectrogram from LIGO/Virgo strain data, determine whether "
    "it contains a gravitational wave signal. If yes, estimate the source "
    "parameters in discrete bins.\n\n"
    "Output strictly as a JSON object with fields:\n"
    '- "detection": "YES" or "NO"\n'
    '- "chirp_mass_bin": predefined bin label (only if YES, else "N/A")\n'
    '- "distance_bin": predefined bin label (only if YES, else "N/A")\n'
    '- "chi_eff_bin": predefined bin label (only if YES, else "N/A")\n\n'
    "Output only the JSON, no other text."
)

SYSTEM_PROMPTS = {
    "detection_only": SYSTEM_PROMPT_DETECTION_ONLY,
    "multitask": SYSTEM_PROMPT_MULTITASK,
}


def build_assistant_target(label: dict, schema: str) -> str:
    """按 schema 生成 assistant 目标 JSON 串（紧凑、字段顺序固定）。"""
    if schema == "detection_only":
        obj = {"detection": label["detection"]}
    else:  # multitask
        obj = {
            "detection": label["detection"],
            "chirp_mass_bin": label["chirp_mass_bin"],
            "distance_bin": label["distance_bin"],
            "chi_eff_bin": label["chi_eff_bin"],
        }
    return json.dumps(obj, ensure_ascii=False)


def resolve_image_field(abs_path: str, image_path_mode: str, image_base: Path) -> str:
    """按模式返回 image 字段值（绝对 or 相对 image_base）。"""
    if image_path_mode == "absolute":
        return abs_path
    # relative：相对 image_base（默认 spectrograms/），当前数据即为 basename
    return os.path.relpath(abs_path, image_base)


def convert_sample(sample: dict, schema: str, system_prompt: str,
                   image_path_mode: str, image_base: Path, user_text: str) -> dict:
    image_field = resolve_image_field(sample["image_path"], image_path_mode, image_base)
    user_content = [{"type": "image", "image": image_field}]
    if user_text:  # 默认空：docs §4.1 规定 user 仅图像、无文字；指令在 system prompt
        user_content.append({"type": "text", "text": user_text})
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": build_assistant_target(sample["label"], schema)},
        ]
    }


def convert_split(in_path: Path, out_path: Path, schema: str, system_prompt: str,
                  image_path_mode: str, image_base: Path, user_text: str) -> Counter:
    if not in_path.exists():
        raise FileNotFoundError(f"缺少 {in_path}，先跑 06/07 生成切分。")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    det_counts: Counter = Counter()
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            det_counts[sample["label"]["detection"]] += 1
            record = convert_sample(sample, schema, system_prompt,
                                    image_path_mode, image_base, user_text)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {in_path.name} → {out_path}  ({n} 条, {dict(det_counts)})")
    return det_counts


def main() -> None:
    ap = argparse.ArgumentParser(description="导出训练器会话格式（E1/E2）")
    ap.add_argument("--schema", choices=list(SCHEMA_TO_SUBDIR), default="detection_only",
                    help="detection_only=E1（默认）/ multitask=E2")
    ap.add_argument("--image-path", choices=("relative", "absolute"), default="relative",
                    help="relative=相对 --image-base（默认，便于跨机器）/ absolute=绝对路径")
    ap.add_argument("--image-base", type=Path, default=SPECTROGRAMS_DIR,
                    help="relative 模式下的基准目录（默认 output/spectrograms/）")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "training_data",
                    help="输出根目录（默认 output/training_data/）")
    ap.add_argument("--user-text", default="",
                    help="user message 文字（默认空，遵循 docs §4.1 仅图像）")
    args = ap.parse_args()

    system_prompt = SYSTEM_PROMPTS[args.schema]
    sub = SCHEMA_TO_SUBDIR[args.schema]
    out_root = args.out_dir / sub

    print(f"[schema={args.schema} → {sub}/]  image-path={args.image_path}  base={args.image_base}")
    total = Counter()
    for split in SPLITS:
        in_path = OUTPUT_DIR / f"dataset_{split}.jsonl"
        out_path = out_root / f"{split}.jsonl"
        total += convert_split(in_path, out_path, args.schema, system_prompt,
                               args.image_path, args.image_base, args.user_text)

    print(f"\n完成：{out_root}/{{train,val,test}}.jsonl")
    print(f"合计 detection 分布：{dict(total)}")
    print(f"system prompt（{args.schema}）:\n---\n{system_prompt}\n---")


if __name__ == "__main__":
    main()
