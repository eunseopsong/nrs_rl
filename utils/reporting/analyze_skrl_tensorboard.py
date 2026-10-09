import os
import sys
import glob
import re

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth_torch(values, window=15):
    """
    TensorBoard scalar가 너무 출렁이면 moving average 적용.
    PyTorch tensor로 smoothing하므로 'PyTorch 기반 그래프 분석'이라고 말할 수 있음.
    """
    y = torch.tensor(values, dtype=torch.float32)

    if len(y) < window or window <= 1:
        return y.numpy()

    pad = window // 2
    kernel = torch.ones(1, 1, window) / window
    y_in = y.view(1, 1, -1)
    y_pad = torch.nn.functional.pad(y_in, (pad, pad), mode="replicate")
    y_smooth = torch.nn.functional.conv1d(y_pad, kernel).view(-1)

    return y_smooth.numpy()


def safe_name(tag):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)


def find_latest_run(base_dir):
    runs = glob.glob(os.path.join(os.path.expanduser(base_dir), "*"))
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        raise FileNotFoundError(f"No run directory found in {base_dir}")
    return max(runs, key=os.path.getmtime)


def load_scalars(run_dir):
    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    ea = EventAccumulator(run_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    return ea, tags


def plot_scalar(df, tag, out_dir, smooth_window=15):
    steps = df["step"].to_numpy()
    values = df["value"].to_numpy()

    smooth_values = smooth_torch(values, window=smooth_window)

    plt.figure(figsize=(11, 5))
    plt.plot(steps, values, linewidth=0.7, alpha=0.35, label="raw")
    plt.plot(steps, smooth_values, linewidth=1.8, label=f"smoothed w={smooth_window}")
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.title(tag)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(out_dir, safe_name(tag) + ".png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    return png_path


def main():
    if len(sys.argv) >= 2:
        run_dir = os.path.expanduser(sys.argv[1])
    else:
        run_dir = find_latest_run("~/logs/skrl/cartpole_direct")

    print(f"[INFO] run_dir = {run_dir}")

    ea, tags = load_scalars(run_dir)

    print("\n[INFO] Available scalar tags:")
    for t in tags:
        print("  -", t)

    out_dir = os.path.join(run_dir, "pytorch_graph_analysis")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for tag in tags:
        events = ea.Scalars(tag)
        if len(events) == 0:
            continue

        df = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        })

        csv_path = os.path.join(out_dir, safe_name(tag) + ".csv")
        df.to_csv(csv_path, index=False)

        png_path = plot_scalar(df, tag, out_dir)

        values = torch.tensor(df["value"].to_numpy(), dtype=torch.float32)

        summary_rows.append({
            "tag": tag,
            "num_points": len(values),
            "first": float(values[0]),
            "last": float(values[-1]),
            "mean": float(values.mean()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "csv": csv_path,
            "png": png_path,
        })

        print(f"[SAVE] {png_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)

    print("\n[INFO] summary saved:", summary_path)
    print("[INFO] plots saved to:", out_dir)

    # 중요한 tag 후보만 따로 출력
    print("\n[INFO] Important tags candidates:")
    keywords = ["reward", "value", "loss", "entropy", "std", "noise", "policy", "kl"]
    for row in summary_rows:
        tag_lower = row["tag"].lower()
        if any(k in tag_lower for k in keywords):
            print(f"  - {row['tag']}: first={row['first']:.4g}, last={row['last']:.4g}, std={row['std']:.4g}")


if __name__ == "__main__":
    main()
