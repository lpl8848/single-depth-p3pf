"""Create the pooled three-sequence TUM RGB-D pilot report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from summarize_results import number, percent, read_csv, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sequences = [
        ("fr1/xyz", "fr1_xyz_gap10"),
        ("fr1/desk", "fr1_desk_gap10"),
        ("fr1/room", "fr1_room_gap10"),
    ]
    pooled: dict[str, list[dict[str, str]]] = {
        "OD-P3Pf, strongest anchor": [],
        "OD-P3Pf, GT-verified anchor": [],
        "PoseLib P4Pf, identical subset": [],
    }
    sequence_rows = []
    for label, prefix in sequences:
        od = read_csv(args.results_dir / f"{prefix}_od.csv")
        p4pf = read_csv(args.results_dir / f"{prefix}_p4pf_gt_verified.csv")
        strongest = [row for row in od if row["anchor_mode"] == "strongest"]
        verified = [row for row in od if row["anchor_mode"] == "verified"]
        pooled["OD-P3Pf, strongest anchor"].extend(strongest)
        pooled["OD-P3Pf, GT-verified anchor"].extend(verified)
        pooled["PoseLib P4Pf, identical subset"].extend(p4pf)
        anchor_correct = np.asarray([
            float(row["anchor_gt_reproj_px"]) < 4.0
            and float(row["anchor_gt_depth_m"])
            < max(0.05, 0.03 * float(row["anchor_query_z_m"]))
            for row in strongest
        ])
        sequence_rows.append((
            label,
            len(strongest),
            len(verified) / len(strongest),
            float(anchor_correct.mean()),
            summarize(strongest)["success_2deg_5cm"],
            summarize(verified)["success_2deg_5cm"],
            summarize(p4pf)["success_2deg_5cm"],
        ))

    summaries = {name: summarize(rows) for name, rows in pooled.items()}
    lines = [
        "# TUM RGB-D real-data experiment",
        "",
        "## Protocol",
        "",
        "Reference-frame RGB-D creates the map-side 3D points. Query RGB supplies the "
        "2D observations, and exactly one query-frame z-depth is exposed to OD-P3Pf. "
        "RGB, depth, and motion-capture poses are associated by timestamp within 30 ms. "
        "Frame separation is 10 RGB entries. SURF matches are scored by 1000 minimal "
        "samples with a 4 px truncated reprojection criterion and no nonlinear refinement.",
        "",
        "## Pooled result",
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
        "## Sequence-level success and anchor reliability",
        "",
        "| Sequence | Frames | Has verified anchor | Strongest anchor correct | "
        "Strongest OD | GT-verified OD | P4Pf (same subset) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, frames, coverage, anchor_precision, strongest_success, verified_success, p4pf_success in sequence_rows:
        lines.append(
            f"| {label} | {frames} | {percent(coverage)} | {percent(anchor_precision)} | "
            f"{percent(strongest_success)} | {percent(verified_success)} | "
            f"{percent(p4pf_success)} |"
        )

    strongest = pooled["OD-P3Pf, strongest anchor"]
    correct = np.asarray([
        float(row["anchor_gt_reproj_px"]) < 4.0
        and float(row["anchor_gt_depth_m"])
        < max(0.05, 0.03 * float(row["anchor_query_z_m"]))
        for row in strongest
    ])
    success = ((number(strongest, "rotation_deg") < 2.0)
               & (number(strongest, "translation_m") < 0.05))
    lines += [
        "",
        "## Interpretation",
        "",
        f"Across all sequences, the strongest descriptor is a valid metric anchor in only "
        f"{percent(float(correct.mean()))} of frames. Conditional strict pose success is "
        f"{percent(float(success[correct].mean()))} with a correct strongest anchor and "
        f"{percent(float(success[~correct].mean()))} otherwise. Thus the real-data result "
        "supports the minimal estimator but rejects descriptor rank alone as an anchor policy.",
        "",
        "The GT-verified-anchor protocol uses ground truth only to isolate estimator feasibility; "
        "it is not a deployable method. Runtime is indicative rather than a cycle-level comparison: "
        "OD-P3Pf is a standalone C++ executable, while P4Pf is called through PoseLib's Python binding.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
