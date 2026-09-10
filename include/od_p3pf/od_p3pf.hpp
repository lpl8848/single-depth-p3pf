#pragma once

#include <array>
#include <vector>

namespace od_p3pf {

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Mat3 {
  // Row-major storage.
  std::array<double, 9> v{};
  double& operator()(int row, int col) { return v[3 * row + col]; }
  double operator()(int row, int col) const { return v[3 * row + col]; }
};

struct Solution {
  Mat3 R;
  Vec3 t;
  double focal = 0.0;
  std::array<double, 3> depth{};
  double constraint_residual = 0.0;
};

// Solves three 3D--2D correspondences with a known metric z-depth for the
// first correspondence and a shared unknown positive focal length.  The
// principal point is assumed to have already been subtracted from pixels.
// Every returned solution has positive focal length and positive depths.
std::vector<Solution> solve(const std::array<Vec3, 3>& world,
                            const std::array<Vec2, 3>& pixels,
                            double known_z);

}  // namespace od_p3pf
