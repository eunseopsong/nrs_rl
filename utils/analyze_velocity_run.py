"""Summarize raw polishing traces without modifying historical logs."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def metrics(path):
    with np.load(path) as data:
        active = data["polishing_active"] > 0
        rate = data["raw_mrr_n_mm_s"][active]
        measured = data["measured_speed_mm_s"][active]
        command = data["commanded_speed_mm_s"][active]
        force = np.abs(data["normal_force_n"][active])
        return {
            "episode": path.parent.name, "samples": int(active.sum()),
            "raw_rate_cv": float(rate.std() / rate.mean()) if rate.mean() else None,
            "mean_rate_n_mm_s": float(rate.mean()),
            "command_mean_mm_s": float(command.mean()), "command_std_mm_s": float(command.std()),
            "measured_speed_std_mm_s": float(measured.std()),
            "tracking_speed_std_mm_s": float((measured - command).std()),
            "force_cv": float(force.std() / force.mean()) if force.mean() else None,
            "zero_rate_fraction": float((rate == 0).mean()),
            "shield_fraction": float(data["safety_shield_active"][active].mean()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.run.glob("ep*/08_control_trace.npz"), key=lambda p: int(p.parent.name[2:]))
    if not paths:
        parser.error("No ep*/08_control_trace.npz files found")
    rows = [metrics(p) for p in paths]
    if args.baseline_trace:
        rows.append(metrics(args.baseline_trace))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "raw_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "raw_metrics.json").write_text(json.dumps(rows, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    trace = np.load(paths[-1])
    active = trace["polishing_active"] > 0
    t = trace["time_s"][active]
    t -= t[0]
    axes[0].plot(t, trace["measured_speed_mm_s"][active], lw=.6, color="#7187ad", label="Measured TCP")
    axes[0].plot(t, trace["commanded_speed_mm_s"][active], lw=1.2, color="#bd5849", label="Applied command")
    axes[1].plot(t, trace["normal_force_n"][active], lw=.7, color="#7187ad")
    axes[2].plot(t, trace["raw_mrr_n_mm_s"][active], lw=.7, color="#7187ad")
    axes[2].axhline(60., color="#bd5849", lw=1, label="Rate target")
    for ax, label in zip(axes, ("Speed [mm/s]", "Normal force [N]", "Raw rate [N mm/s]")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[2].legend()
    axes[2].set_xlabel("Polishing time [s]")
    fig.suptitle(f"{args.run.name} / {paths[-1].parent.name} (training exploration included)")
    fig.tight_layout()
    fig.savefig(args.output / "last_episode_raw.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
