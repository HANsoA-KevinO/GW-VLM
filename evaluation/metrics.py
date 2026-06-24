"""
检测指标：accuracy / precision / recall / F1 / ROC-AUC / PR-AUC / 混淆矩阵。
另含从模型自由文本输出里解析 detection（YES/NO）的鲁棒解析器。
"""
import json
import re
from typing import Optional


def parse_detection(text: Optional[str]) -> Optional[str]:
    """从模型输出抽取 detection。先试 JSON，再正则兜底；抽不到返回 None。"""
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = str(json.loads(m.group(0)).get("detection", "")).strip().upper()
            if d in ("YES", "NO"):
                return d
        except Exception:
            pass
    t = text.strip().upper()
    has_yes, has_no = "YES" in t, "NO" in t
    if has_yes and not has_no:
        return "YES"
    if has_no and not has_yes:
        return "NO"
    return None


def _to_binary(label: str) -> int:
    return 1 if str(label).strip().upper() == "YES" else 0


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """y_true / y_pred: 'YES'/'NO' 列表；y_score: P(YES) 可选，给了才算 ROC/PR AUC。"""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
        classification_report, roc_auc_score, average_precision_score,
    )
    yt = [_to_binary(x) for x in y_true]
    yp = [_to_binary(x) for x in y_pred]

    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    out = {
        "n": len(yt),
        "accuracy": accuracy_score(yt, yp),
        "precision_YES": p,
        "recall_YES": r,
        "f1_YES": f1,
        "confusion_matrix": {
            "labels": ["NO", "YES"],
            "matrix": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),  # 行=真实, 列=预测
        },
        "per_class": classification_report(
            yt, yp, labels=[0, 1], target_names=["NO", "YES"],
            output_dict=True, zero_division=0,
        ),
    }
    if y_score is not None and len(set(yt)) > 1:
        try:
            out["roc_auc"] = roc_auc_score(yt, y_score)
            out["pr_auc"] = average_precision_score(yt, y_score)
        except Exception:
            out["roc_auc"] = out["pr_auc"] = None
    return out


def save_confusion_png(matrix, path, labels=("NO", "YES")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
