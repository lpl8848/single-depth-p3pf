"""PoseLib P4Pf RANSAC baseline on the TUM OD-P3Pf export format."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import poselib


@dataclass
class Frame:
    frame_id: int
    focal_gt: float
    rotation_gt: np.ndarray
    translation_gt: np.ndarray
    world: np.ndarray
    pixels: np.ndarray


def load_frames(path: Path) -> list[Frame]:
    raw = [line.strip() for line in path.read_text().splitlines()]
    lines = [line for line in raw if line and not line.startswith("#")]
    frames: list[Frame] = []
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor].split()
        if header[0] != "FRAME":
            raise ValueError(f"Expected FRAME at line {cursor}: {lines[cursor]}")
        frame_id, count = int(header[1]), int(header[2])
        cursor += 1
        intrinsics = lines[cursor].split()
        focal_gt = float(intrinsics[1])
        cursor += 1
        pose_values = np.asarray(lines[cursor].split()[1:], dtype=float).reshape(3, 4)
        cursor += 1
        data = np.asarray(
            [[float(value) for value in lines[cursor + i].split()] for i in range(count)]
        )
        cursor += count
        frames.append(
            Frame(
                frame_id,
                focal_gt,
                pose_values[:, :3],
                pose_values[:, 3],
                data[:, :3],
                data[:, 3:5],
            )
        )
    return frames


def score_model(frame: Frame, rotation: np.ndarray, translation: np.ndarray,
                focal: float, threshold: float) -> tuple[int, float]:
    camera = frame.world @ rotation.T + translation
    valid = camera[:, 2] > 1e-9
    predicted = np.full_like(frame.pixels, np.inf)
    predicted[valid] = focal * camera[valid, :2] / camera[valid, 2, None]
    squared = np.sum((predicted - frame.pixels) ** 2, axis=1)
    threshold_squared = threshold * threshold
    return int(np.count_nonzero(squared < threshold_squared)), float(
        np.minimum(squared, threshold_squared).sum()
    )


def rotation_error(rotation: np.ndarray, truth: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation @ truth.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def run_frame(frame: Frame, iterations: int, threshold: float,
              rng: np.random.Generator) -> dict[str, float | int]:
    best = None
    hypotheses = 0
    start = time.perf_counter()
    for _ in range(iterations):
        sample = rng.choice(frame.world.shape[0], 4, replace=False)
        poses, focals = poselib.p4pf(frame.pixels[sample], frame.world[sample], True)
        hypotheses += len(poses)
        for pose, focal in zip(poses, focals):
            if not np.isfinite(focal) or focal <= 0.0:
                continue
            rotation = np.asarray(pose.R)
            translation = np.asarray(pose.t)
            inliers, msac = score_model(frame, rotation, translation, focal, threshold)
            if best is None or inliers > best[0] or (inliers == best[0] and msac < best[1]):
                best = (inliers, msac, rotation, translation, float(focal))
    runtime_ms = 1e3 * (time.perf_counter() - start)
    if best is None:
        return {
            "valid": 0,
            "inliers": 0,
            "hypotheses": hypotheses,
            "runtime_ms": runtime_ms,
            "rotation_deg": np.nan,
            "translation_m": np.nan,
            "focal_relative": np.nan,
        }
    inliers, _, rotation, translation, focal = best
    return {
        "valid": 1,
        "inliers": inliers,
        "hypotheses": hypotheses,
        "runtime_ms": runtime_ms,
        "rotation_deg": rotation_error(rotation, frame.rotation_gt),
        "translation_m": float(np.linalg.norm(translation - frame.translation_gt)),
        "focal_relative": abs(focal - frame.focal_gt) / frame.focal_gt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument(
        "--frame-ids-from-od",
        type=Path,
        help="Restrict evaluation to frame IDs marked anchor_mode=verified in this OD CSV.",
    )
    args = parser.parse_args()

    frames = load_frames(args.input)
    if args.frame_ids_from_od is not None:
        with args.frame_ids_from_od.open(newline="") as handle:
            allowed = {
                int(row["frame"])
                for row in csv.DictReader(handle)
                if row["anchor_mode"] == "verified"
            }
        frames = [frame for frame in frames if frame.frame_id in allowed]
        missing = allowed.difference(frame.frame_id for frame in frames)
        if missing:
            raise ValueError(f"OD frame IDs absent from export: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame", "method", "matches", "valid", "inliers", "hypotheses",
        "runtime_ms", "rotation_deg", "translation_m", "focal_relative",
    ]
    rng = np.random.default_rng(args.seed)
    rows = []
    for index, frame in enumerate(frames, start=1):
        result = run_frame(frame, args.iterations, args.threshold, rng)
        row = {"frame": frame.frame_id, "method": "p4pf", "matches": len(frame.world)}
        row.update(result)
        rows.append(row)
        if index % 10 == 0 or index == len(frames):
            print(f"P4Pf: {index}/{len(frames)} frames", flush=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
