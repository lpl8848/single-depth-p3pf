# Single-Depth P3P with Unknown Focal Length (OD-P3Pf)

This repository contains the compact solver and the core experiment drivers for
absolute pose from three 3D--2D correspondences, one unknown focal length, and
one camera-axis metric depth. Pairwise distance invariance reduces the problem
to a fixed degree-six polynomial in `f^-2`.

The release intentionally contains only the code needed for the main numerical
claims. Symbolic derivations, root-search utilities, one-off validation scripts,
paper sources, generated figures, result caches, and datasets are excluded.

## Contents

- `include/od_p3pf/od_p3pf.hpp`, `src/od_p3pf.cpp`: dependency-free C++17
  implementation of the direct sextic solver.
- `src/benchmark.cpp`: exact six-physical-solution witness, regression tests,
  noise/degeneracy sweeps, and runtime measurements.
- `src/od_p3pf_gb.cpp`, `src/same_problem_benchmark.cpp`: optional generated
  solver for the identical equations and its matched benchmark.
- `python/solver.py`: readable NumPy reference implementation.
- `python/ransac_value.py`: sample-budget calculation and fixed-anchor RANSAC
  simulation.
- `experiments/tum/`: TUM RGB-D export, OD-P3Pf/P4Pf experiment drivers, and
  scripts for pooled, per-sequence, and candidate-anchor summaries.

## Build and test

The main solver has no external dependency.

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On a single-configuration generator, omit `-C Release`. The regression test
runs the six-solution witness followed by short noise-free and noisy trials.

Run the complete synthetic benchmark with:

```powershell
./build/Release/od_p3pf_benchmark.exe
./build/Release/od_p3pf_benchmark.exe --sweep
./build/Release/od_p3pf_benchmark.exe --stress
```

Adjust the executable path to `./build/od_p3pf_benchmark` on Linux or macOS.

## Optional same-system generated baseline

The generated baseline uses Eigen 3.3 or newer. Point CMake to an installed
Eigen package and enable it explicitly:

```powershell
cmake -S . -B build-gb -DOD_P3PF_BUILD_GB_BASELINE=ON
cmake --build build-gb --config Release
ctest --test-dir build-gb -C Release --output-on-failure
```

This produces `od_p3pf_same_problem_benchmark`, which compares the direct
sextic with a 16-by-22 elimination template and 6-by-6 action matrix on exactly
the same equations.

## Python reference and RANSAC experiment

```powershell
python -m pip install -r requirements.txt
python python/solver.py --trials 1000
python python/ransac_value.py --trials 50 --inliers 0.2 0.3 0.5
```

The analytic table reports the 95% sample budgets for P4Pf, a resampled metric
anchor plus two visual correspondences, and a once-selected fixed anchor.

## TUM RGB-D experiment

The dataset is not redistributed. Download the TUM RGB-D sequences separately,
then use MATLAB with Computer Vision Toolbox to export associated frame pairs:

```matlab
export_tum_od('path/to/rgbd_dataset_freiburg1_xyz', ...
              'results/fr1_xyz_gap10.txt', inf, 10)
```

Run the strongest/verified-anchor diagnostic and the GT-free candidate-anchor
experiment:

```powershell
./build/Release/od_p3pf_tum.exe results/fr1_xyz_gap10.txt results/fr1_xyz_gap10_od.csv 1000 20260908
./build/Release/od_p3pf_tum.exe results/fr1_xyz_gap10.txt results/fr1_xyz_gap10_candidate_anchor.csv 1000 20260908 candidate
```

Run the PoseLib P4Pf baseline on the identical exported pairs:

```powershell
python experiments/tum/run_p4pf_ransac.py results/fr1_xyz_gap10.txt results/fr1_xyz_gap10_p4pf.csv --iterations 1000 --seed 20260908
```

Repeat for `fr1/desk` and `fr1/room`, retaining the filename prefixes expected
by the summary scripts. Then generate the pooled P90/per-sequence report and the
candidate-anchor report:

```powershell
python experiments/tum/summarize_all_results.py results results/tum_summary.md
python experiments/tum/summarize_candidate_anchors.py results results/candidate_anchor.md results/candidate_anchor.csv results/candidate_anchor.png
```

All stochastic drivers expose their seed and use fixed defaults. Generated
outputs belong under `results/` and are ignored by Git.
