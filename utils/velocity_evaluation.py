"""Quality gates for independent raw-signal velocity policy rollouts."""
import math


def compare_surfaces(adaptive, constant):
    """Require CV improvement at comparable mean depth on the identical ROI."""
    if adaptive["roi_cells"] != constant["roi_cells"] or adaptive["roi_area_mm2"] != constant["roi_area_mm2"]:
        raise ValueError("Preston comparisons require the same fixed equal-area ROI")
    a, c = adaptive["spatial_depth_cv"], constant["spatial_depth_cv"]
    am, cm = adaptive["mean_depth_over_k"], constant["mean_depth_over_k"]
    valid = all(x is not None and math.isfinite(x) and x >= 0 for x in (a, c, am, cm))
    gain = 1 - a / c if valid and c > 0 else None
    mean_ratio = am / cm if valid and cm > 0 else None
    coverage_ok = adaptive["zero_depth_fraction"] <= constant["zero_depth_fraction"] + 1e-12
    return {
        "spatial_cv_improvement_fraction": gain, "mean_depth_ratio": mean_ratio,
        "uncovered_area_not_increased": coverage_ok,
        "passed": bool(gain is not None and gain > 0 and mean_ratio is not None
                       and .95 <= mean_ratio <= 1.05 and coverage_ok),
    }


def compare_rotation_surfaces(adaptive, constant, minimum_improvement=0.):
    """Compare normalized h/(K*omega); never mix it with the TCP h/K model."""
    if not 0 <= minimum_improvement < 1:
        raise ValueError('Minimum depth CV improvement must be in [0, 1)')
    normalized = []
    for metrics in (adaptive, constant):
        normalized.append({**metrics, 'mean_depth_over_k': metrics['mean_depth_over_k_omega']})
    comparison = compare_surfaces(*normalized)
    gain = comparison['spatial_cv_improvement_fraction']
    comparison['passed'] &= gain is not None and gain > minimum_improvement
    comparison['minimum_depth_cv_improvement_fraction'] = minimum_improvement
    comparison['depth_normalization'] = 'K*omega; constant RPM, rotation-dominated assumption'
    return comparison


def compare_rollouts(adaptive, constant, prior=None, minimum_improvement=0.0):
    if not 0.0 <= minimum_improvement < 1.0:
        raise ValueError("minimum CV improvement must be in [0, 1)")

    def versus(baseline):
        for metrics in (adaptive, baseline):
            for name in ("processing_rate_cv", "processing_samples", "processing_mean_rate"):
                value = metrics[name]
                if not math.isfinite(value) or value < 0 or (name != "processing_rate_cv" and value == 0):
                    raise ValueError(f"Invalid evaluation metric {name}={value}")
        cv = baseline["processing_rate_cv"]
        improvement = 1.0 - adaptive["processing_rate_cv"] / cv if cv > 0 else 0.0
        duration = adaptive["processing_samples"] / baseline["processing_samples"]
        rate = adaptive["processing_mean_rate"] / baseline["processing_mean_rate"]
        return improvement, duration, rate

    improvement, duration, rate = versus(constant)
    safe = adaptive["shield_fraction"] == 0.0 and adaptive.get("fault_fraction", 0.0) == 0.0
    passed = bool(adaptive["completed"] and constant["completed"] and safe
                  and improvement > minimum_improvement and duration <= 1.05 and rate >= .95)
    comparison = {
        "raw_cv_improvement_fraction": improvement,
        "processing_duration_ratio": duration,
        "mean_rate_ratio": rate,
        "minimum_cv_improvement_fraction": minimum_improvement,
    }
    if prior is not None:
        gain, prior_duration, prior_rate = versus(prior)
        comparison.update({"cv_improvement_over_process_prior": gain,
                           "duration_ratio_to_process_prior": prior_duration,
                           "mean_rate_ratio_to_process_prior": prior_rate})
        passed = bool(passed and prior["completed"] and gain > 0.0
                      and prior_duration <= 1.05 and prior_rate >= .95)
    comparison["passed"] = passed
    return comparison
