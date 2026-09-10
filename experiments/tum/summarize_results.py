"""Summarize paired OD-P3Pf and PoseLib P4Pf real-data RANSAC runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    rotation = number(rows, "rotation_deg")
    translation = number(rows, "translation_m")
    focal = number(rows, "focal_relative")
    runtime = number(rows, "runtime_ms")
    inliers = number(rows, "inliers")
    matches = number(rows, "matches")
    return {
        "frames": len(rows),
        "rot_med": np.nanmedian(rotation),
        "rot_p90": np.nanpercentile(rotation, 90),
        "trans_med": np.nanmedian(translation),
        "trans_p90": np.nanpercentile(translation, 90),
        "focal_med": np.nanmedian(focal),
        "focal_p90": np.nanpercentile(focal, 90),
        "inlier_med": np.nanmedian(inliers / matches),
        "runtime_med": np.nanmedian(runtime),
        "success_2deg_5cm": np.nanmean((rotation < 2.0) & (translation < 0.05)),
        "success_5deg_10cm": np.nanmean((rotation < 5.0) & (translation < 0.10)),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("od", type=Path)
    parser.add_argument("p4pf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    od_rows = read_csv(args.od)
    p4pf_rows = read_csv(args.p4pf)
    methods = {
        "OD-P3Pf, strongest anchor": [row for row in od_rows if row["anchor_mode"] == "strongest"],
        "OD-P3Pf, verified anchor": [row for row in od_rows if row["anchor_mode"] == "verified"],
        "P4Pf": p4pf_rows,
    }
    summaries = {name: summarize(rows) for name, rows in methods.items()}

    strongest = methods["OD-P3Pf, strongest anchor"]
    anchor_correct = np.asarray(
        [float(row["anchor_gt_reproj_px"]) < 4.0 and float(row["anchor_gt_depth_m"]) <
         max(0.05, 0.03 * float(row["anchor_query_z_m"])) for row in strongest]
    )
    strongest_success = (
        (number(strongest, "rotation_deg") < 2.0)
        & (number(strongest, "translation_m") < 0.05)
    )
    anchor_precision = float(anchor_correct.mean())
    success_if_anchor = float(strongest_success[anchor_correct].mean()) if anchor_correct.any() else np.nan
    success_if_bad_anchor = float(strongest_success[~anchor_correct].mean()) if (~anchor_correct).any() else np.nan

    verified = methods["OD-P3Pf, verified anchor"]
    p4pf = methods["P4Pf"]
    verified_by_frame = {int(row["frame"]): row for row in verified}
    p4pf_by_frame = {int(row["frame"]): row for row in p4pf}
    common = sorted(set(verified_by_frame) & set(p4pf_by_frame))
    od_better_rotation = np.mean([
        float(verified_by_frame[i]["rotation_deg"]) < float(p4pf_by_frame[i]["rotation_deg"])
        for i in common
    ])
    od_better_translation = np.mean([
        float(verified_by_frame[i]["translation_m"]) < float(p4pf_by_frame[i]["translation_m"])
        for i in common
    ])
    od_better_focal = np.mean([
        float(verified_by_frame[i]["focal_relative"]) < float(p4pf_by_frame[i]["focal_relative"])
        for i in common
    ])

    lines = [
        f"# Real-data pilot: {args.label or args.od.stem.replace('_od', '')}",
        "",
        "All methods use 1000 minimal samples and a 4 px MSAC/RANSAC threshold. "
        "OD-P3Pf uses one query-frame metric z-depth; P4Pf uses no depth.",
        "",
        "| Method | Frames | Median rot. | P90 rot. | Median trans. | P90 trans. | "
        "Median focal | P90 focal | 2deg/5cm | 5deg/10cm | Median time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summaries.items():
        lines.append(
            f"| {name} | {result['frames']} | {result['rot_med']:.3f} deg | "
            f"{result['rot_p90']:.3f} deg | {100*result['trans_med']:.2f} cm | "
            f"{100*result['trans_p90']:.2f} cm | {percent(result['focal_med'])} | "
            f"{percent(result['focal_p90'])} | {percent(result['success_2deg_5cm'])} | "
            f"{percent(result['success_5deg_10cm'])} | {result['runtime_med']:.2f} ms |"
        )
    lines += [
        "",
        "## Anchor diagnosis",
        "",
        f"- The strongest descriptor match is a geometrically/depth-consistent anchor in "
        f"{percent(anchor_precision)} of frames.",
        f"- Strict pose success (2 deg, 5 cm) conditional on a correct strongest anchor: "
        f"{percent(success_if_anchor)}.",
        f"- The same success conditional on an incorrect strongest anchor: "
        f"{percent(success_if_bad_anchor)}.",
        "",
        "## Paired comparison: verified OD anchor versus P4Pf",
        "",
        f"- OD-P3Pf has lower rotation error in {percent(float(od_better_rotation))} of paired frames.",
        f"- OD-P3Pf has lower translation error in {percent(float(od_better_translation))} of paired frames.",
        f"- OD-P3Pf has lower focal error in {percent(float(od_better_focal))} of paired frames.",
        "",
        "The verified-anchor result isolates estimator feasibility; it is not a deployable anchor-selection method. "
        "The strongest-anchor result is the practical one-anchor stress test.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
