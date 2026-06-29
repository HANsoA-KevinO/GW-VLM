"""
按 split_key 把 dataset.jsonl 切成 train / val / test (60/20/20)。

关键约束：同一 group（事件 ID / 注入 ID / glitch ID）的所有样本
必须落在同一切分集合，避免数据泄露。具体规则：
- real_pos + real_neg_off：用 event_name 作为 group_key（pos/neg 同一 strain 段，必须同切分）
- inject_pos：用 split_key（"inject_xxx"）
- glitch_neg：用 split_key（"glitch_xxx"）

切分先按 group_kind 分层（real / inject / glitch），每类内独立 shuffle 切 60/20/20，再合并。

输出：output/dataset_train.jsonl / dataset_val.jsonl / dataset_test.jsonl
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_DIR


DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
TRAIN_PATH = OUTPUT_DIR / "dataset_train.jsonl"
VAL_PATH = OUTPUT_DIR / "dataset_val.jsonl"
TEST_PATH = OUTPUT_DIR / "dataset_test.jsonl"
SPLIT_REPORT_PATH = OUTPUT_DIR / "split_report.txt"

SPLIT_RATIOS = (0.80, 0.10, 0.10)
RANDOM_SEED = 42


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"先跑 06_build_dataset.py 生成 {DATASET_PATH}")
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def split_keys(keys: list[str], rng: random.Random) -> tuple[set, set, set]:
    """打乱 keys，按 60/20/20 切。"""
    keys = list(keys)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    train = set(keys[:n_train])
    val = set(keys[n_train:n_train + n_val])
    test = set(keys[n_train + n_val:])
    return train, val, test


def _group_kind(source_type: str) -> str:
    """real_pos 和 real_neg_off 同事件同切分 → 同一 group_kind 'real'。"""
    if source_type in ("real_pos", "real_neg_off"):
        return "real"
    if source_type == "inject_pos":
        return "inject"
    if source_type == "glitch_neg":
        return "glitch"
    return "other"


def _group_key(sample: dict) -> str:
    """real 用 event_name；inject/glitch 用各自 split_key。"""
    if sample["source_type"] in ("real_pos", "real_neg_off"):
        return sample["event_name"]
    return sample["split_key"]


def split_dataset(samples: list[dict]) -> tuple[list, list, list]:
    rng = random.Random(RANDOM_SEED)
    by_kind: dict[str, set] = defaultdict(set)
    for s in samples:
        by_kind[_group_kind(s["source_type"])].add(_group_key(s))

    key_to_split: dict[tuple[str, str], str] = {}
    for kind, keys in by_kind.items():
        # 合成样本(注入/glitch)只进 train:① 防"合成 vs 真实"捷径 ② val/test 全真实 = 量真实泛化。
        # real 先于 inject 处理(dataset 里 real 在前)→ real 的切分与无注入时完全一致(seed 42)。
        if kind in ("inject", "glitch"):
            for k in keys:
                key_to_split[(kind, k)] = "train"
            continue
        train_k, val_k, test_k = split_keys(sorted(keys), rng)
        for k in train_k:
            key_to_split[(kind, k)] = "train"
        for k in val_k:
            key_to_split[(kind, k)] = "val"
        for k in test_k:
            key_to_split[(kind, k)] = "test"

    train, val, test = [], [], []
    for s in samples:
        bucket = key_to_split[(_group_kind(s["source_type"]), _group_key(s))]
        if bucket == "train":
            train.append(s)
        elif bucket == "val":
            val.append(s)
        else:
            test.append(s)

    return train, val, test


def write_split(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def report_split(train: list, val: list, test: list) -> str:
    def stats(samples: list[dict]) -> str:
        n = len(samples)
        src = Counter(s["source_type"] for s in samples)
        det = Counter(s["label"]["detection"] for s in samples)
        keys = len(set((s["source_type"], s["split_key"]) for s in samples))
        return (
            f"  total: {n} | unique split_keys: {keys}\n"
            f"  source_type: {dict(src)}\n"
            f"  detection: {dict(det)}"
        )

    lines = [
        f"Random seed: {RANDOM_SEED}",
        f"Split ratios: train={SPLIT_RATIOS[0]:.0%} val={SPLIT_RATIOS[1]:.0%} test={SPLIT_RATIOS[2]:.0%}",
        "",
        "Train:",
        stats(train),
        "",
        "Val:",
        stats(val),
        "",
        "Test:",
        stats(test),
    ]
    return "\n".join(lines)


def main() -> None:
    samples = load_dataset()
    train, val, test = split_dataset(samples)

    write_split(train, TRAIN_PATH)
    write_split(val, VAL_PATH)
    write_split(test, TEST_PATH)

    report = report_split(train, val, test)
    SPLIT_REPORT_PATH.write_text(report + "\n")
    print(report)
    print()
    print(f"Written: {TRAIN_PATH.name}, {VAL_PATH.name}, {TEST_PATH.name}")
    print(f"Report:  {SPLIT_REPORT_PATH}")


if __name__ == "__main__":
    main()
