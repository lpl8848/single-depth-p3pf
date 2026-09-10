#include "od_p3pf/od_p3pf.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace od_p3pf {
namespace {

using Poly = std::vector<double>;  // Ascending coefficient order.
using Complex = std::complex<double>;

constexpr double kTiny = 1e-14;

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
  return n > kTiny ? (1.0 / n) * a : Vec3{};
}

Vec3 mul(const Mat3& a, const Vec3& x) {
  return {a(0, 0) * x.x + a(0, 1) * x.y + a(0, 2) * x.z,
          a(1, 0) * x.x + a(1, 1) * x.y + a(1, 2) * x.z,
          a(2, 0) * x.x + a(2, 1) * x.y + a(2, 2) * x.z};
}

Poly trim(Poly p, double relative = 1e-11) {
  double scale = 1.0;
  for (double value : p) scale = std::max(scale, std::abs(value));
  while (p.size() > 1 && std::abs(p.back()) <= relative * scale) p.pop_back();
  return p;
}

Poly add(const Poly& a, const Poly& b, double b_scale = 1.0) {
  Poly out(std::max(a.size(), b.size()), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) out[i] += a[i];
  for (std::size_t i = 0; i < b.size(); ++i) out[i] += b_scale * b[i];
  return out;
}

Poly scale(const Poly& a, double s) {
  Poly out = a;
  for (double& value : out) value *= s;
  return out;
}

Poly mul(const Poly& a, const Poly& b) {
  Poly out(a.size() + b.size() - 1, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
  return out;
}

double eval(const Poly& p, double x) {
  double value = 0.0;
  for (auto it = p.rbegin(); it != p.rend(); ++it) value = value * x + *it;
  return value;
}

Complex eval(const Poly& p, Complex x) {
  Complex value = 0.0;
  for (auto it = p.rbegin(); it != p.rend(); ++it) value = value * x + *it;
  return value;
}

Poly derivative(const Poly& p) {
  if (p.size() <= 1) return {0.0};
  Poly out(p.size() - 1);
  for (std::size_t i = 1; i < p.size(); ++i) out[i - 1] = i * p[i];
  return out;
}

std::vector<double> positive_quadratic_roots(double a, double b, double c) {
  std::vector<double> roots;
  const double scale_value = std::max({1.0, std::abs(a), std::abs(b), std::abs(c)});
  if (std::abs(a) <= 1e-13 * scale_value) {
    if (std::abs(b) > 1e-13 * scale_value) {
      const double root = -c / b;
      if (root > 0.0 && std::isfinite(root)) roots.push_back(root);
    }
    return roots;
  }
  double discriminant = b * b - 4.0 * a * c;
  if (discriminant < -2e-11 * scale_value * scale_value) return roots;
  discriminant = std::max(0.0, discriminant);
  const double square_root = std::sqrt(discriminant);
  // This form avoids cancellation when |b| is close to sqrt(discriminant).
  const double q = -0.5 * (b + std::copysign(square_root, b));
  if (std::abs(q) > kTiny) {
    const double first = q / a;
    const double second = c / q;
    if (first > 0.0 && std::isfinite(first)) roots.push_back(first);
    if (second > 0.0 && std::isfinite(second) &&
        std::abs(second - first) > 1e-10 * (1.0 + std::abs(first)))
      roots.push_back(second);
  } else {
    const double root = -b / (2.0 * a);
    if (root > 0.0 && std::isfinite(root)) roots.push_back(root);
  }
  return roots;
}

// Fixed-degree Aberth--Ehrlich polynomial root finder.  Coefficient-based
// variable scaling is important here because the physical variable k=1/f^2
// can be several orders of magnitude below one even after pixel normalization.
std::vector<Complex> aberth_roots(Poly p) {
  p = trim(std::move(p), 1e-13);
  const int degree = static_cast<int>(p.size()) - 1;
  if (degree <= 0) return {};
  if (degree == 1) return {Complex(-p[0] / p[1], 0.0)};

  std::vector<double> candidate_scales;
  const double lead = std::abs(p.back());
  for (int i = 0; i < degree; ++i) {
    if (std::abs(p[i]) > kTiny * lead) {
      candidate_scales.push_back(
          std::pow(std::abs(p[i] / p.back()), 1.0 / (degree - i)));
    }
  }
  double variable_scale = 1.0;
  if (!candidate_scales.empty()) {
    std::sort(candidate_scales.begin(), candidate_scales.end());
    variable_scale = candidate_scales[candidate_scales.size() / 2];
    variable_scale = std::clamp(variable_scale, 1e-10, 1e10);
  }

  Poly q(p.size());
  double power = 1.0;
  for (int i = 0; i <= degree; ++i) {
    q[i] = p[i] * power;
    power *= variable_scale;
  }
  const double qlead = q.back();
  for (double& value : q) value /= qlead;
  const Poly dq = derivative(q);

  double radius = 1.0;
  for (int i = 0; i < degree; ++i) radius = std::max(radius, 1.0 + std::abs(q[i]));
  radius = std::min(radius, 1e4);
  constexpr double pi = 3.141592653589793238462643383279502884;
  std::vector<Complex> roots(degree);
  for (int i = 0; i < degree; ++i) {
    const double angle = 2.0 * pi * (i + 0.37) / degree;
    roots[i] = std::polar(radius, angle);
  }

  for (int iteration = 0; iteration < 120; ++iteration) {
    bool converged = true;
    const auto old = roots;
    for (int i = 0; i < degree; ++i) {
      const Complex value = eval(q, old[i]);
      const Complex slope = eval(dq, old[i]);
      if (std::abs(slope) < 1e-30) {
        converged = false;
        continue;
      }
      const Complex newton = value / slope;
      Complex repulsion = 0.0;
      for (int j = 0; j < degree; ++j) {
        if (i != j && std::abs(old[i] - old[j]) > 1e-30)
          repulsion += 1.0 / (old[i] - old[j]);
      }
      Complex denominator = 1.0 - newton * repulsion;
      if (std::abs(denominator) < 1e-12) denominator = 1.0;
      const Complex step = newton / denominator;
      roots[i] = old[i] - step;
      if (std::abs(step) > 2e-13 * (1.0 + std::abs(roots[i]))) converged = false;
    }
    if (converged) break;
  }

  for (Complex& root : roots) root *= variable_scale;
  return roots;
}

Complex laguerre_single(const std::vector<Complex>& coefficients, int degree,
                        Complex initial) {
  constexpr int kMaxIterations = 120;
  constexpr double fractions[9] = {0.0, 0.5, 0.25, 0.75, 0.13,
                                    0.38, 0.62, 0.88, 1.0};
  Complex x = initial;
  for (int iteration = 1; iteration <= kMaxIterations; ++iteration) {
    Complex value = coefficients[degree];
    Complex first = 0.0;
    Complex half_second = 0.0;
    double error = std::abs(value);
    const double abs_x = std::abs(x);
    for (int j = degree - 1; j >= 0; --j) {
      half_second = x * half_second + first;
      first = x * first + value;
      value = x * value + coefficients[j];
      error = std::abs(value) + abs_x * error;
    }
    if (std::abs(value) <= 2e-15 * error) return x;
    const Complex g = first / value;
    const Complex h = g * g - 2.0 * half_second / value;
    const Complex square_root =
        std::sqrt(static_cast<double>(degree - 1) *
                  (static_cast<double>(degree) * h - g * g));
    const Complex plus = g + square_root;
    const Complex minus = g - square_root;
    const Complex denominator = std::abs(plus) > std::abs(minus) ? plus : minus;
    Complex step;
    if (std::abs(denominator) > 1e-30) {
      step = static_cast<double>(degree) / denominator;
    } else {
      step = std::polar(1.0 + abs_x, static_cast<double>(iteration));
    }
    const Complex next = x - step;
    if (next == x || std::abs(step) <= 2e-14 * (1.0 + std::abs(next))) return next;
    // Fractional steps every ten iterations break rare limit cycles.
    if (iteration % 10 == 0)
      x -= fractions[(iteration / 10) % 8 + 1] * step;
    else
      x = next;
  }
  return x;
}

std::vector<Complex> laguerre_roots(Poly p) {
  p = trim(std::move(p), 1e-13);
  const int degree = static_cast<int>(p.size()) - 1;
  if (degree <= 0) return {};

  // Use the same coefficient-derived variable scale as the simultaneous
  // solver, then deflate one root at a time in the normalized variable.
  std::vector<double> candidate_scales;
  const double lead = std::abs(p.back());
  for (int i = 0; i < degree; ++i) {
    if (std::abs(p[i]) > kTiny * lead)
      candidate_scales.push_back(
          std::pow(std::abs(p[i] / p.back()), 1.0 / (degree - i)));
  }
  double variable_scale = 1.0;
  if (!candidate_scales.empty()) {
    std::sort(candidate_scales.begin(), candidate_scales.end());
    variable_scale = std::clamp(candidate_scales[candidate_scales.size() / 2],
                                1e-10, 1e10);
  }
  std::vector<Complex> original(degree + 1);
  double power = 1.0;
  for (int i = 0; i <= degree; ++i) {
    original[i] = p[i] * power;
    power *= variable_scale;
  }
  for (Complex& value : original) value /= original.back();
  std::vector<Complex> deflated = original;
  std::vector<Complex> roots(degree);
  for (int current = degree; current >= 1; --current) {
    Complex root = laguerre_single(deflated, current, Complex(0.0, 0.0));
    if (std::abs(root.imag()) <= 2e-13 * (1.0 + std::abs(root.real())))
      root = Complex(root.real(), 0.0);
    roots[current - 1] = root;
    Complex accumulator = deflated[current];
    for (int j = current - 1; j >= 0; --j) {
      const Complex saved = deflated[j];
      deflated[j] = accumulator;
      accumulator = root * accumulator + saved;
    }
    deflated.resize(current);
  }
  // Deflation loses a few digits for clustered roots; polish against the
  // original polynomial before undoing the variable scaling.
  for (Complex& root : roots) {
    root = laguerre_single(original, degree, root) * variable_scale;
  }
  return roots;
}

std::vector<Complex> polynomial_roots(const Poly& p) {
  auto roots = laguerre_roots(p);
  // Aberth is retained as a deterministic safety net if Laguerre ever emits a
  // non-finite value.  This branch is not taken on the benchmark corpus.
  for (const Complex& root : roots) {
    if (!std::isfinite(root.real()) || !std::isfinite(root.imag()))
      return aberth_roots(p);
  }
  return roots;
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
    const double fv[3] = {f[0].x, f[1].x, f[2].x};
    const double gv[3] = {f[0].y, f[1].y, f[2].y};
    const double hv[3] = {f[0].z, f[1].z, f[2].z};
    const double* component = row == 0 ? fv : (row == 1 ? gv : hv);
    for (int col = 0; col < 3; ++col) {
      // E=[e1 e2 e3], hence row `col` of E is formed from the
      // corresponding coordinate of all three basis vectors.
      const double ec[3] = {e[0].x, e[1].x, e[2].x};
      const double fc[3] = {e[0].y, e[1].y, e[2].y};
      const double gc[3] = {e[0].z, e[1].z, e[2].z};
      const double* ecoord = col == 0 ? ec : (col == 1 ? fc : gc);
      (*rotation)(row, col) = component[0] * ecoord[0] +
                              component[1] * ecoord[1] +
                              component[2] * ecoord[2];
    }
  }
  *translation = camera[0] - mul(*rotation, world[0]);
  return true;
}

double squared_distance(const Vec3& a, const Vec3& b) {
  const Vec3 d = a - b;
  return dot(d, d);
}

}  // namespace

std::vector<Solution> solve(const std::array<Vec3, 3>& world,
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
  const double l12 = squared_distance(world[0], world[1]);
  const double l13 = squared_distance(world[0], world[2]);
  const double l23 = squared_distance(world[1], world[2]);
  if (l12 < 1e-16 || l13 < 1e-16 || l23 < 1e-16) return solutions;

  const Poly a{1.0, dot(p[1], p[1])};
  const Poly b{-2.0 * z, -2.0 * z * dot(p[0], p[1])};
  const Poly c{z * z - l12, z * z * dot(p[0], p[0])};
  const Poly d{1.0, dot(p[2], p[2])};
  const Poly e{-2.0 * z, -2.0 * z * dot(p[0], p[2])};
  const Poly f{z * z - l13, z * z * dot(p[0], p[0])};
  const Poly coupling{-2.0, -2.0 * dot(p[1], p[2])};

  const Poly p0 = scale(add(add(c, f), Poly{l23}), -1.0);
  const Poly p1 = scale(b, -1.0);
  const Poly p2 = scale(e, -1.0);
  const Poly p3 = coupling;

  Poly h0 = add(add(mul(d, mul(p0, p0)),
                         scale(mul(e, mul(p0, p2)), -1.0)),
                    mul(f, mul(p2, p2)));
  Poly h1 = add(add(scale(mul(d, mul(p0, p1)), 2.0),
                         scale(mul(e, add(mul(p0, p3), mul(p1, p2))), -1.0)),
                    scale(mul(f, mul(p2, p3)), 2.0));
  Poly h2 = add(add(mul(d, mul(p1, p1)),
                         scale(mul(e, mul(p1, p3)), -1.0)),
                    mul(f, mul(p3, p3)));

  // Remove the common k^2 F_x determinant term symbolically.  This is the
  // structural degree-8-to-6 cancellation; enforcing it before root finding
  // prevents two numerically tiny leading coefficients from making huge fake
  // roots.
  const double det13 = p[0].x * p[2].y - p[0].y * p[2].x;
  const Poly cancellation{0.0, 0.0, 4.0 * z * z * det13 * det13};
  h0 = trim(add(h0, mul(cancellation, c), -1.0));
  h1 = trim(add(h1, mul(cancellation, b), -1.0));
  h2 = trim(add(h2, mul(cancellation, a), -1.0));

  const Poly ah0_ch2 = add(mul(a, h0), mul(c, h2), -1.0);
  const Poly ah1_bh2 = add(mul(a, h1), mul(b, h2), -1.0);
  const Poly bh0_ch1 = add(mul(b, h0), mul(c, h1), -1.0);
  const Poly resultant =
      trim(add(mul(ah0_ch2, ah0_ch2), mul(ah1_bh2, bh0_ch1), -1.0), 1e-10);
  if (resultant.size() < 2 || resultant.size() > 7) return solutions;

  const Poly resultant_derivative = derivative(resultant);
  std::vector<double> real_roots;
  for (Complex root : polynomial_roots(resultant)) {
    if (std::abs(root.imag()) > 2e-6 * (1.0 + std::abs(root.real()))) continue;
    double k = root.real();
    // Real Newton polishing is cheap and materially improves the recovered
    // depths when two focal roots are close.
    for (int iteration = 0; iteration < 8; ++iteration) {
      const double slope = eval(resultant_derivative, k);
      if (std::abs(slope) < 1e-20) break;
      const double step = eval(resultant, k) / slope;
      k -= step;
      if (std::abs(step) < 1e-13 * (1.0 + std::abs(k))) break;
    }
    if (!(k > 0.0) || !std::isfinite(k)) continue;
    bool duplicate = false;
    for (double prior : real_roots)
      duplicate |= std::abs(k - prior) < 1e-7 * (1.0 + std::abs(k));
    if (!duplicate) real_roots.push_back(k);
  }

  const double equation_scale = std::max({1.0, l12, l13, l23, z * z});
  for (double k : real_roots) {
    const double av = eval(a, k), bv = eval(b, k), cv = eval(c, k);
    const double dv = eval(d, k), ev = eval(e, k), fv = eval(f, k);
    const double h0v = eval(h0, k), h1v = eval(h1, k), h2v = eval(h2, k);
    const double u = av * h0v - cv * h2v;
    const double v = av * h1v - bv * h2v;
    std::vector<std::pair<double, double>> depth_candidates;
    if (std::abs(v) >= 1e-13 * equation_scale) {
      const double x = -u / v;
      const double denominator = eval(p2, k) + eval(p3, k) * x;
      if (x > 0.0 && std::isfinite(x) &&
          std::abs(denominator) >= 1e-13 * equation_scale) {
        const double y = -(eval(p0, k) + eval(p1, k) * x) / denominator;
        if (y > 0.0 && std::isfinite(y)) depth_candidates.emplace_back(x, y);
      }
    }
    // Near a vanishing linear subresultant, direct division is poorly
    // conditioned.  Pairing the two stable quadratic roots is a fixed-cost
    // fallback (at most four pairs) and also guards close sextic roots.
    const auto x_roots = positive_quadratic_roots(av, bv, cv);
    const auto y_roots = positive_quadratic_roots(dv, ev, fv);
    for (double x : x_roots)
      for (double y : y_roots) depth_candidates.emplace_back(x, y);

    double x = 0.0, y = 0.0;
    double algebraic_residual = std::numeric_limits<double>::infinity();
    for (const auto& candidate : depth_candidates) {
      const double xc = candidate.first, yc = candidate.second;
      const double f12 = av * xc * xc + bv * xc + cv;
      const double f13 = dv * yc * yc + ev * yc + fv;
      const double g = eval(p0, k) + eval(p1, k) * xc + eval(p2, k) * yc +
                       eval(p3, k) * xc * yc;
      const double residual =
          std::max({std::abs(f12), std::abs(f13), std::abs(g)}) / equation_scale;
      if (residual < algebraic_residual) {
        algebraic_residual = residual;
        x = xc;
        y = yc;
      }
    }
    if (algebraic_residual > 2e-5) continue;
    const double inverse_focal_normalized = std::sqrt(k);
    const std::array<Vec3, 3> camera{{
        {z * inverse_focal_normalized * p[0].x,
         z * inverse_focal_normalized * p[0].y, z},
        {x * inverse_focal_normalized * p[1].x,
         x * inverse_focal_normalized * p[1].y, x},
        {y * inverse_focal_normalized * p[2].x,
         y * inverse_focal_normalized * p[2].y, y},
    }};
    Solution solution;
    if (!recover_pose(world, camera, &solution.R, &solution.t)) continue;
    solution.focal = pixel_scale / inverse_focal_normalized;
    solution.depth = {z, x, y};
    solution.constraint_residual = algebraic_residual;
    solutions.push_back(solution);
  }
  std::sort(solutions.begin(), solutions.end(),
            [](const Solution& a, const Solution& b) {
              return a.constraint_residual < b.constraint_residual;
            });
  return solutions;
}

}  // namespace od_p3pf
