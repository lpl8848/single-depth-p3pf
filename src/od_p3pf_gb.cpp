#include "od_p3pf/od_p3pf_gb.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace od_p3pf {
namespace {

double dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 operator*(double s, const Vec3& a) { return {s * a.x, s * a.y, s * a.z}; }
double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}
Vec3 normalized(const Vec3& a) {
  const double n = norm(a);
  return n > 1e-14 ? (1.0 / n) * a : Vec3{};
}
Vec3 mul(const Mat3& a, const Vec3& x) {
  return {a(0, 0) * x.x + a(0, 1) * x.y + a(0, 2) * x.z,
          a(1, 0) * x.x + a(1, 1) * x.y + a(1, 2) * x.z,
          a(2, 0) * x.x + a(2, 1) * x.y + a(2, 2) * x.z};
}
double squared_distance(const Vec3& a, const Vec3& b) { return dot(a - b, a - b); }

void set_linear(Eigen::MatrixXd* matrix, int index_one_based, double value) {
  const int index = index_one_based - 1;
  (*matrix)(index % matrix->rows(), index / matrix->rows()) = value;
}
double get_linear(const Eigen::MatrixXd& matrix, int index_one_based) {
  const int index = index_one_based - 1;
  return matrix(index % matrix.rows(), index / matrix.rows());
}

void rref(Eigen::MatrixXd* matrix) {
  int pivot_row = 0;
  const double scale = std::max(1.0, matrix->cwiseAbs().rowwise().sum().maxCoeff());
  const double tolerance = std::max(matrix->rows(), matrix->cols()) *
                           std::numeric_limits<double>::epsilon() * scale;
  for (int col = 0; col < matrix->cols() && pivot_row < matrix->rows(); ++col) {
    int best = pivot_row;
    double magnitude = 0.0;
    for (int row = pivot_row; row < matrix->rows(); ++row) {
      if (std::abs((*matrix)(row, col)) > magnitude) {
        magnitude = std::abs((*matrix)(row, col));
        best = row;
      }
    }
    if (magnitude <= tolerance) continue;
    if (best != pivot_row) matrix->row(best).swap(matrix->row(pivot_row));
    matrix->row(pivot_row) /= (*matrix)(pivot_row, col);
    for (int row = 0; row < matrix->rows(); ++row) {
      if (row == pivot_row) continue;
      const double factor = (*matrix)(row, col);
      if (std::abs(factor) > tolerance)
        matrix->row(row) -= factor * matrix->row(pivot_row);
    }
    ++pivot_row;
  }
}

bool recover_pose(const std::array<Vec3, 3>& world,
                  const std::array<Vec3, 3>& camera, Mat3* rotation,
                  Vec3* translation) {
  const Vec3 wx = world[1] - world[0];
  const Vec3 wy = world[2] - world[0];
  const Vec3 cx = camera[1] - camera[0];
  const Vec3 cy = camera[2] - camera[0];
  const Vec3 e1 = normalized(wx);
  const Vec3 f1 = normalized(cx);
  const Vec3 e2raw = wy - dot(wy, e1) * e1;
  const Vec3 f2raw = cy - dot(cy, f1) * f1;
  if (norm(e2raw) < 1e-11 || norm(f2raw) < 1e-11) return false;
  const Vec3 e2 = normalized(e2raw);
  const Vec3 f2 = normalized(f2raw);
  const Vec3 e3 = cross(e1, e2);
  const Vec3 f3 = cross(f1, f2);
  const Vec3 e[3] = {e1, e2, e3};
  const Vec3 f[3] = {f1, f2, f3};
  for (int row = 0; row < 3; ++row) {
    const double fc[3] = {row == 0 ? f[0].x : (row == 1 ? f[0].y : f[0].z),
                          row == 0 ? f[1].x : (row == 1 ? f[1].y : f[1].z),
                          row == 0 ? f[2].x : (row == 1 ? f[2].y : f[2].z)};
    for (int col = 0; col < 3; ++col) {
      const double ec[3] = {col == 0 ? e[0].x : (col == 1 ? e[0].y : e[0].z),
                            col == 0 ? e[1].x : (col == 1 ? e[1].y : e[1].z),
                            col == 0 ? e[2].x : (col == 1 ? e[2].y : e[2].z)};
      (*rotation)(row, col) = fc[0] * ec[0] + fc[1] * ec[1] + fc[2] * ec[2];
    }
  }
  *translation = camera[0] - mul(*rotation, world[0]);
  return true;
}

}  // namespace

std::vector<Solution> solve_gb(const std::array<Vec3, 3>& world,
                               const std::array<Vec2, 3>& pixels,
                               double known_z) {
  std::vector<Solution> solutions;
  if (!(known_z > 0.0) || !std::isfinite(known_z)) return solutions;

  double pixel_energy = 0.0;
  for (const Vec2& point : pixels) pixel_energy += dot(point, point);
  const double pixel_scale = std::sqrt(pixel_energy / 6.0);
  if (!(pixel_scale > 1e-12) || !std::isfinite(pixel_scale)) return solutions;
  std::array<Vec2, 3> p{};
  for (int i = 0; i < 3; ++i)
    p[i] = {pixels[i].x / pixel_scale, pixels[i].y / pixel_scale};

  const double z = known_z;
  const double L12 = squared_distance(world[0], world[1]);
  const double L13 = squared_distance(world[0], world[2]);
  const double L23 = squared_distance(world[1], world[2]);
  if (L12 < 1e-16 || L13 < 1e-16 || L23 < 1e-16) return solutions;

  std::array<double, 20> c{};
  c[0] = 1.0;
  c[1] = dot(p[1], p[1]);
  c[2] = -2.0 * z * dot(p[0], p[1]);
  c[3] = -2.0 * z;
  c[4] = z * z * dot(p[0], p[0]);
  c[5] = z * z - L12;
  c[6] = 1.0;
  c[7] = dot(p[2], p[2]);
  c[8] = -2.0 * z * dot(p[0], p[2]);
  c[9] = -2.0 * z;
  c[10] = z * z * dot(p[0], p[0]);
  c[11] = z * z - L13;
  c[12] = -2.0 * dot(p[1], p[2]);
  c[13] = -2.0;
  c[14] = 2.0 * z * dot(p[0], p[2]);
  c[15] = 2.0 * z * dot(p[0], p[1]);
  c[16] = 2.0 * z;
  c[17] = 2.0 * z;
  c[18] = -2.0 * z * z * dot(p[0], p[0]);
  c[19] = L12 + L13 - L23 - 2.0 * z * z;

  Eigen::MatrixXd M1 = Eigen::MatrixXd::Zero(3, 12);
  const int m1_index[20] = {10, 1, 19, 25, 31, 34, 17, 8, 23, 29,
                            32, 35, 6, 15, 24, 21, 30, 27, 33, 36};
  for (int i = 0; i < 20; ++i) set_linear(&M1, m1_index[i], c[i]);
  rref(&M1);

  Eigen::MatrixXd M2 = Eigen::MatrixXd::Zero(12, 26);
  const int m2_cols[12] = {12, 13, 14, 17, 18, 19, 20, 21, 23, 24, 25, 26};
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 12; ++col) M2(9 + row, m2_cols[col] - 1) = M1(row, col);
  const int m2_dst[20][3] = {{1,14,51},{16,29,66},{31,44,81},{85,98,135},{100,113,150},
      {115,128,165},{133,146,171},{136,149,174},{148,161,186},{151,164,189},
      {193,206,231},{196,209,234},{208,221,246},{211,224,249},{229,242,255},
      {232,245,258},{235,248,261},{265,278,291},{268,281,294},{271,284,297}};
  const int m2_src[20] = {1,5,9,10,14,18,19,20,23,24,25,26,29,30,31,32,33,34,35,36};
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 3; ++j) set_linear(&M2, m2_dst[i][j], get_linear(M1, m2_src[i]));
  rref(&M2);

  Eigen::MatrixXd M3 = Eigen::MatrixXd::Zero(18, 29);
  const int m3_cols[26] = {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};
  for (int row = 0; row < 12; ++row)
    for (int col = 0; col < 26; ++col) M3(6 + row, m3_cols[col] - 1) = M2(row, col);
  const int m3_dst[18][3] = {{1,20,75},{22,41,96},{181,200,255},{199,218,273},{202,221,276},
      {220,239,294},{253,272,309},{256,275,312},{271,290,327},{274,293,330},
      {343,362,399},{346,365,402},{361,380,417},{364,383,420},{397,416,435},
      {400,419,438},{451,470,489},{454,473,492}};
  const int m3_src[18] = {104,117,200,212,213,225,236,237,248,249,272,273,284,285,296,297,308,309};
  for (int i = 0; i < 18; ++i)
    for (int j = 0; j < 3; ++j) set_linear(&M3, m3_dst[i][j], get_linear(M2, m3_src[i]));
  rref(&M3);

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(16, 22);
  const int m_rows[13] = {2,5,6,8,9,10,11,12,13,14,15,16,18};
  const int m_cols[22] = {2,5,6,8,9,10,12,13,15,16,17,18,19,21,22,23,24,25,26,27,28,29};
  for (int row = 0; row < 13; ++row)
    for (int col = 0; col < 22; ++col) M(3 + row, col) = M3(m_rows[row] - 1, m_cols[col] - 1);
  const int m_dst[7][3] = {{1,18,35},{17,50,67},{33,66,83},{97,130,147},
                           {113,146,163},{145,178,195},{209,242,259}};
  const int m_src[7] = {378,414,432,468,486,504,522};
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 3; ++j) set_linear(&M, m_dst[i][j], get_linear(M3, m_src[i]));
  rref(&M);

  Eigen::Matrix<double, 6, 6> action = Eigen::Matrix<double, 6, 6>::Zero();
  action(0, 2) = 1.0;
  action(1, 5) = 1.0;
  const int action_rows[4] = {15, 14, 13, 11};
  const int action_cols[6] = {22, 21, 20, 19, 18, 17};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j) action(2 + i, j) = -M(action_rows[i] - 1, action_cols[j] - 1);

  Eigen::ComplexEigenSolver<Eigen::Matrix<double, 6, 6>> eigen(action, true);
  if (eigen.info() != Eigen::Success) return solutions;
  const auto vectors = eigen.eigenvectors();
  for (int col = 0; col < 6; ++col) {
    const std::complex<double> denominator = vectors(0, col);
    if (std::abs(denominator) < 1e-13) continue;
    const std::complex<double> xc = vectors(3, col) / denominator;
    const std::complex<double> yc = vectors(2, col) / denominator;
    const std::complex<double> kc = vectors(1, col) / denominator;
    const double imag_scale = 1.0 + std::max({std::abs(xc.real()), std::abs(yc.real()), std::abs(kc.real())});
    if (std::max({std::abs(xc.imag()), std::abs(yc.imag()), std::abs(kc.imag())}) > 2e-6 * imag_scale)
      continue;
    const double x = xc.real(), y = yc.real(), k = kc.real();
    if (!(x > 0.0 && y > 0.0 && k > 0.0) ||
        !std::isfinite(x + y + k)) continue;

    const double fx = (x-z)*(x-z) + k*((x*p[1].x-z*p[0].x)*(x*p[1].x-z*p[0].x) +
        (x*p[1].y-z*p[0].y)*(x*p[1].y-z*p[0].y)) - L12;
    const double fy = (y-z)*(y-z) + k*((y*p[2].x-z*p[0].x)*(y*p[2].x-z*p[0].x) +
        (y*p[2].y-z*p[0].y)*(y*p[2].y-z*p[0].y)) - L13;
    const double g = L12+L13-L23 - 2.0*((x-z)*(y-z) + k*((x*p[1].x-z*p[0].x)*(y*p[2].x-z*p[0].x) +
        (x*p[1].y-z*p[0].y)*(y*p[2].y-z*p[0].y)));
    const double residual_scale = std::max({1.0, L12, L13, L23});
    const double residual = std::sqrt((fx*fx + fy*fy + g*g) / 3.0) / residual_scale;
    if (!(residual < 5e-4)) continue;

    const double inverse_focal = std::sqrt(k);
    std::array<Vec3, 3> camera{{
        {z * inverse_focal * p[0].x, z * inverse_focal * p[0].y, z},
        {x * inverse_focal * p[1].x, x * inverse_focal * p[1].y, x},
        {y * inverse_focal * p[2].x, y * inverse_focal * p[2].y, y}}};
    Solution solution;
    if (!recover_pose(world, camera, &solution.R, &solution.t)) continue;
    solution.focal = pixel_scale / inverse_focal;
    solution.depth = {z, x, y};
    solution.constraint_residual = residual;
    solutions.push_back(solution);
  }
  return solutions;
}

}  // namespace od_p3pf
