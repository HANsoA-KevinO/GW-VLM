"""
从 Trainer 的 trainer_state.json 画训练/验证损失随步数(epoch)的曲线。

数据源：SFTTrainer 把完整 log_history(每 logging_steps 的 train loss + 每 epoch 的 eval_loss)
存进 <run_dir>/trainer_state.json 或 <run_dir>/checkpoint-*/trainer_state.json。

用法：
  python evaluation/plot_loss.py output/runs/e1_gemma4_e4b           # 自动找最新 trainer_state.json
  python evaluation/plot_loss.py <path/to/trainer_state.json> --out curve.png
"""
import argparse
import json
from pathlib import Path


def find_trainer_state(p: Path) -> Path:
    if p.is_file():
        return p
    root = p / "trainer_state.json"
    if root.exists():
        return root
    cks = sorted(p.glob("checkpoint-*/trainer_state.json"),
                 key=lambda x: int(x.parent.name.split("-")[1]))
    if cks:
        return cks[-1]
    raise FileNotFoundError(f"在 {p} 下找不到 trainer_state.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="画训练/验证损失曲线")
    ap.add_argument("path", type=Path, help="run 目录 或 trainer_state.json")
    ap.add_argument("--out", default=None, help="输出 PNG 路径")
    args = ap.parse_args()

    ts = find_trainer_state(args.path)
    state = json.loads(ts.read_text())
    hist = state.get("log_history", [])

    train = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    evals = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    epochs = state.get("epoch")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_name = ts.parent.name if ts.parent.name.startswith("checkpoint") else args.path.name
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train:
        ax.plot(*zip(*train), label="train loss", color="tab:blue", lw=1.4)
    if evals:
        ax.plot(*zip(*evals), label="eval loss", color="tab:red", marker="o", ms=5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"GW-VLM E1 loss — {run_name} ({epochs:.0f} epochs)" if epochs else f"loss — {run_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        (args.path if args.path.is_dir() else ts.parent) / "loss_curve.png")
    fig.savefig(out, dpi=120)
    print(f"训练点数={len(train)}  验证点数={len(evals)}")
    if train:
        print(f"train loss: 首 {train[0][1]:.4f} → 末 {train[-1][1]:.4f}")
    if evals:
        print(f"eval  loss: {[round(v,4) for _, v in evals]}")
    print(f"已保存曲线: {out}")


if __name__ == "__main__":
    main()
