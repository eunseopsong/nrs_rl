#!/usr/bin/env python3
"""Reintegrate paired raw control traces onto the same Preston surface ROI."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = _REPO_ROOT
TASK = ROOT / "source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl"
spec = importlib.util.spec_from_file_location("preston_surface", TASK / "utils/preston_surface.py")
preston = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preston)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--trajectory", type=Path, default=TASK / "datasets/cmd_continue9D_flat.h5")
    parser.add_argument("--tool-diameter-mm", type=float, default=30.)
    parser.add_argument("--contact-diameter-mm", type=float)
    parser.add_argument("--cell-size-mm", type=float, default=2.)
    parser.add_argument("--pressure-profile", choices=("uniform", "hertz"), default="uniform")
    parser.add_argument("--velocity-model", choices=("tcp", "rotating_disk"), default="tcp")
    parser.add_argument("--spindle-rpm", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.trajectory) as stream:
        path = stream["position"][:, :3]
    summary_file = args.evaluation / "summary.json"
    source = json.loads(summary_file.read_text())
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    recorded_length = source["path_length_mm"]
    if recorded_length > arc[-1] + .1:
        raise ValueError("Reference path is shorter than the recorded evaluation")
    end = int(np.argmin(np.abs(arc - recorded_length))) + 1
    if abs(arc[end-1] - recorded_length) > .1:
        raise ValueError("Reference path length does not match recorded evaluation")
    surface = preston.PlanarPrestonSurface(
        path[:end], tool_diameter_mm=args.tool_diameter_mm,
        contact_diameter_mm=args.contact_diameter_mm, cell_size_mm=args.cell_size_mm,
        pressure_profile=args.pressure_profile, velocity_model=args.velocity_model,
        spindle_rpm=args.spindle_rpm)
    output = args.output or args.evaluation / ("preston_surface_" + args.velocity_model)
    output.mkdir(parents=True, exist_ok=True)
    result = {"model": surface.metadata, "source_evaluation": str(args.evaluation.resolve())}
    maps = {}
    for mode in ("adaptive", "constant", "process_prior"):
        trace_file = args.evaluation / f"{mode}_trace.npz"
        if not trace_file.exists():
            continue
        with np.load(trace_file) as trace:
            result[mode] = surface.integrate_trace(trace)
        surface.save(output / f"{mode}_surface.npz")
        maps[mode] = surface.depth.copy()
    if "adaptive" in result and "constant" in result:
        a, c = result["adaptive"], result["constant"]
        result["comparison"] = {
            "spatial_cv_improvement_fraction": (1 - a["spatial_depth_cv"] / c["spatial_depth_cv"])
                if a["spatial_depth_cv"] is not None and c["spatial_depth_cv"] else None,
            "mean_depth_ratio": a["mean_depth_over_k"] / c["mean_depth_over_k"]
                if c["mean_depth_over_k"] else None,
            "physical_depth_validated": False,
        }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.5), squeeze=False,
                             layout="constrained")
    maximum = max(float(x[surface.roi].max()) for x in maps.values())
    for ax, (mode, depth) in zip(axes[0], maps.items()):
        z = np.where(surface.roi, depth, np.nan)
        im = ax.imshow(z, origin="lower", extent=(surface.u[0], surface.u[-1], surface.v[0], surface.v[-1]),
                       vmin=0, vmax=maximum, cmap="viridis")
        cv = result[mode]["spatial_depth_cv"]
        ax.set_title(f"{mode}: depth CV {cv:.3f}" if cv is not None else f"{mode}: no removal")
        ax.set_xlabel("Plane u [mm]")
        ax.set_ylabel("Plane v [mm]")
    fig.colorbar(im, ax=list(axes[0]), label="h/K (uncalibrated Preston depth)")
    fig.savefig(output / "surface_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
