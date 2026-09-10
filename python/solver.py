"""Direct sextic solver for P3P+f with one known metric z-depth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import numpy as np
from numpy.polynomial import Polynomial as Poly


@dataclass
class Solution:
    R: np.ndarray
    t: np.ndarray
    focal: float
    z: np.ndarray
    residual: float


def _kabsch(x_world, y_camera):
    xm = x_world.mean(axis=0)
    ym = y_camera.mean(axis=0)
    u, _, vt = np.linalg.svd((x_world - xm).T @ (y_camera - ym))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    t = ym - r @ xm
    return r, t


def _trim(poly: Poly, relative=1e-10):
    coef = np.asarray(poly.coef, dtype=float)
    scale = max(1.0, float(np.max(np.abs(coef))))
    while len(coef) > 1 and abs(coef[-1]) <= relative * scale:
        coef = coef[:-1]
    return Poly(coef)


def _quadratic_roots(a, b, c, tol=1e-10):
    if abs(a) <= tol * max(1.0, abs(b), abs(c)):
        return [] if abs(b) <= tol else [-c / b]
    roots = np.roots([a, b, c])
    return [float(r.real) for r in roots if abs(r.imag) <= 1e-7 * max(1.0, abs(r.real))]


def solve(x_world, pixels, known_z, tol=1e-8):
    x_world = np.asarray(x_world, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    if x_world.shape != (3, 3) or pixels.shape != (3, 2):
        raise ValueError("expected three 3D points and three 2D pixels")
    if known_z <= 0:
        return []

    # Pixel normalization only rescales focal and k.
    pixel_scale = math.sqrt(float(np.mean(pixels * pixels)))
    if pixel_scale <= 1e-12:
        return []
    p = pixels / pixel_scale
    p1, p2, p3 = p
    z1 = float(known_z)
    l12 = float(np.sum((x_world[0] - x_world[1]) ** 2))
    l13 = float(np.sum((x_world[0] - x_world[2]) ** 2))
    l23 = float(np.sum((x_world[1] - x_world[2]) ** 2))

    affine = lambda constant, slope: Poly([float(constant), float(slope)])
    a = affine(1.0, p2 @ p2)
    b = affine(-2.0 * z1, -2.0 * z1 * (p1 @ p2))
    c = affine(z1 * z1 - l12, z1 * z1 * (p1 @ p1))
    d = affine(1.0, p3 @ p3)
    e = affine(-2.0 * z1, -2.0 * z1 * (p1 @ p3))
    f = affine(z1 * z1 - l13, z1 * z1 * (p1 @ p1))
    coupling = affine(-2.0, -2.0 * (p2 @ p3))

    p0 = -(c + f + Poly([l23]))
    p1c = -b
    p2c = -e
    p3c = coupling

    h0 = d * p0 * p0 - e * p0 * p2c + f * p2c * p2c
    h1 = 2.0 * d * p0 * p1c - e * (p0 * p3c + p1c * p2c) + 2.0 * f * p2c * p3c
    h2 = d * p1c * p1c - e * p1c * p3c + f * p3c * p3c

    # Enforce the exact determinant cancellation before floating-point root
    # finding.  Expanding the apparent octic and hoping that its two leading
    # coefficients cancel numerically can create spurious huge roots.
    det13 = p1[0] * p3[1] - p1[1] * p3[0]
    cancellation = Poly([0.0, 0.0, 4.0 * z1 * z1 * det13 * det13])
    h0 = _trim(h0 - cancellation * c)
    h1 = _trim(h1 - cancellation * b)
    h2 = _trim(h2 - cancellation * a)
    resultant = _trim((a * h0 - c * h2) ** 2 - (a * h1 - b * h2) * (b * h0 - c * h1))

    solutions = []
    for root in resultant.roots():
        if abs(root.imag) > 1e-7 * max(1.0, abs(root.real)):
            continue
        k = float(root.real)
        if k <= 0:
            continue
        av, bv, cv = a(k), b(k), c(k)
        dv, ev, fv = d(k), e(k), f(k)
        for x in _quadratic_roots(av, bv, cv):
            for y in _quadratic_roots(dv, ev, fv):
                if x <= 0 or y <= 0:
                    continue
                gv = p0(k) + p1c(k) * x + p2c(k) * y + p3c(k) * x * y
                scale = max(1.0, abs(l12), abs(l13), abs(l23))
                if abs(gv) > 5e-5 * scale:
                    continue
                inv_f_normalized = math.sqrt(k)
                y_camera = np.array(
                    [
                        z1 * np.array([inv_f_normalized * p1[0], inv_f_normalized * p1[1], 1.0]),
                        x * np.array([inv_f_normalized * p2[0], inv_f_normalized * p2[1], 1.0]),
                        y * np.array([inv_f_normalized * p3[0], inv_f_normalized * p3[1], 1.0]),
                    ]
                )
                r, t = _kabsch(x_world, y_camera)
                fit = y_camera - (x_world @ r.T + t)
                residual = math.sqrt(float(np.mean(np.sum(fit * fit, axis=1))))
                solutions.append(
                    Solution(r, t, pixel_scale / inv_f_normalized, np.array([z1, x, y]), residual)
                )
    solutions.sort(key=lambda s: s.residual)
    return solutions


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def _rotation_error(r1, r2):
    value = np.clip((np.trace(r1 @ r2.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def run_trials(trials, seed):
    rng = np.random.default_rng(seed)
    failures = 0
    rotation_errors = []
    translation_errors = []
    focal_errors = []
    counts = []
    runtimes = []
    for _ in range(trials):
        focal = rng.uniform(400.0, 1800.0)
        y_camera = np.column_stack(
            (
                rng.uniform(-2.0, 2.0, 3),
                rng.uniform(-1.5, 1.5, 3),
                rng.uniform(3.0, 10.0, 3),
            )
        )
        r_gt = _random_rotation(rng)
        t_gt = rng.uniform(-2.0, 2.0, 3)
        x_world = (y_camera - t_gt) @ r_gt
        pixels = focal * y_camera[:, :2] / y_camera[:, 2:3]
        start = time.perf_counter()
        candidates = solve(x_world, pixels, y_camera[0, 2])
        runtimes.append(time.perf_counter() - start)
        counts.append(len(candidates))
        if not candidates:
            failures += 1
            continue
        best = min(
            candidates,
            key=lambda s: _rotation_error(s.R, r_gt) + np.linalg.norm(s.t - t_gt),
        )
        rotation_errors.append(_rotation_error(best.R, r_gt))
        translation_errors.append(float(np.linalg.norm(best.t - t_gt)))
        focal_errors.append(abs(best.focal - focal) / focal)

    median = lambda values: float(np.median(values)) if values else float("nan")
    return {
        "trials": trials,
        "failures": failures,
        "median_solutions": median(counts),
        "median_rotation_deg": median(rotation_errors),
        "median_translation": median(translation_errors),
        "median_focal_relative": median(focal_errors),
        "median_runtime_us": 1e6 * median(runtimes),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    print(run_trials(args.trials, args.seed))
