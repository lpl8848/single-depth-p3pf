"""RANSAC sample complexity for fixed and resampled metric anchors."""

from __future__ import annotations

import argparse
import math

import numpy as np

from solver import _random_rotation, _rotation_error, solve


def required_iterations(success_probability: float, good_sample_probability: float) -> int:
    if good_sample_probability >= 1.0:
        return 1
    if good_sample_probability <= 0.0:
        return math.inf
    return math.ceil(
        math.log1p(-success_probability) / math.log1p(-good_sample_probability)
    )


def required_fixed_anchor_iterations(
    success_probability: float, visual_inlier: float, anchor_inlier: float
) -> int | float:
    """Unconditional budget when the one anchor is drawn once and then held fixed.

    The total success probability is
    w_d [1 - (1 - w^2)^M], so it is capped by w_d independently of M.
    """
    if success_probability >= anchor_inlier or anchor_inlier <= 0.0:
        return math.inf
    return required_iterations(success_probability / anchor_inlier, visual_inlier**2)


def iteration_table(success_probability: float = 0.95, anchor_inlier: float = 0.99):
    rows = []
    for visual_inlier in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        p4 = visual_inlier**4
        p_resampled_anchor = anchor_inlier * visual_inlier**2
        fixed = required_fixed_anchor_iterations(
            success_probability, visual_inlier, anchor_inlier
        )
        rows.append(
            {
                "visual_inlier": visual_inlier,
                "P4Pf": required_iterations(success_probability, p4),
                "resampled_anchor_plus_two": required_iterations(
                    success_probability, p_resampled_anchor
                ),
                "fixed_anchor_plus_two": fixed,
                "P4_over_fixed_anchor": required_iterations(success_probability, p4)
                / fixed,
            }
        )
    return rows


def _project(x_world, r, t, focal):
    y = x_world @ r.T + t
    uv = focal * y[:, :2] / y[:, 2:3]
    return uv, y[:, 2]


def _best_ransac_model(
    x_world,
    pixels,
    known_z,
    iterations,
    threshold,
    rng,
):
    indices = np.arange(1, len(x_world))
    best = None
    best_key = (-1, math.inf)
    for _ in range(iterations):
        picked = rng.choice(indices, size=2, replace=False)
        sample = np.r_[0, picked]
        for candidate in solve(x_world[sample], pixels[sample], known_z):
            predicted, depths = _project(
                x_world, candidate.R, candidate.t, candidate.focal
            )
            errors = np.linalg.norm(predicted - pixels, axis=1)
            errors[depths <= 0] = math.inf
            inliers = errors <= threshold
            count = int(np.count_nonzero(inliers))
            truncated = float(np.sum(np.minimum(errors, threshold) ** 2))
            key = (count, -truncated)
            if key > best_key:
                best_key = key
                best = candidate
    return best


def simulate(
    trials: int,
    visual_inlier: float,
    pixel_noise: float,
    depth_rel_noise: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    anchor_inlier = 1.0
    iterations = required_iterations(0.99, anchor_inlier * visual_inlier**2)
    successes = 0
    rotation_errors = []
    translation_errors = []
    focal_errors = []

    for _ in range(trials):
        count = 60
        focal = rng.uniform(600.0, 1400.0)
        y_camera = np.column_stack(
            (
                rng.uniform(-2.5, 2.5, count),
                rng.uniform(-1.8, 1.8, count),
                rng.uniform(4.0, 12.0, count),
            )
        )
        r_gt = _random_rotation(rng)
        t_gt = rng.uniform(-2.0, 2.0, 3)
        x_world = (y_camera - t_gt) @ r_gt
        pixels, _ = _project(x_world, r_gt, t_gt, focal)
        pixels += rng.normal(scale=pixel_noise, size=pixels.shape)

        # Point 0 is the single range/depth-tagged correspondence and is kept
        # correct.  The remaining visual associations follow the requested
        # inlier rate.
        non_anchor = np.arange(1, count)
        inlier_count = max(2, int(round(visual_inlier * len(non_anchor))))
        rng.shuffle(non_anchor)
        outliers = non_anchor[inlier_count:]
        pixels[outliers, 0] = rng.uniform(-900.0, 900.0, len(outliers))
        pixels[outliers, 1] = rng.uniform(-650.0, 650.0, len(outliers))
        known_z = y_camera[0, 2] * (1.0 + rng.normal(scale=depth_rel_noise))

        model = _best_ransac_model(
            x_world, pixels, known_z, iterations, 4.0, rng
        )
        if model is None:
            continue
        rerr = _rotation_error(model.R, r_gt)
        terr = float(np.linalg.norm(model.t - t_gt))
        ferr = abs(model.focal - focal) / focal
        rotation_errors.append(rerr)
        translation_errors.append(terr)
        focal_errors.append(ferr)
        if rerr < 3.0 and terr < 0.5 and ferr < 0.1:
            successes += 1

    def quantiles(values):
        if not values:
            return {"median": math.nan, "p90": math.nan}
        return {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }

    return {
        "trials": trials,
        "visual_inlier": visual_inlier,
        "iterations": iterations,
        "success_rate": successes / trials,
        "rotation_deg": quantiles(rotation_errors),
        "translation": quantiles(translation_errors),
        "focal_relative": quantiles(focal_errors),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--pixel-noise", type=float, default=1.0)
    parser.add_argument("--depth-rel-noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--inliers", type=float, nargs="*", default=[0.2, 0.3, 0.5])
    args = parser.parse_args()

    print("iteration table")
    for row in iteration_table():
        print(row)
    print("solver simulation")
    for offset, inlier in enumerate(args.inliers):
        print(
            simulate(
                args.trials,
                inlier,
                args.pixel_noise,
                args.depth_rel_noise,
                args.seed + offset,
            )
        )
