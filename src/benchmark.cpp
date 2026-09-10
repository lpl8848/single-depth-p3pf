#include "od_p3pf/od_p3pf.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using od_p3pf::Mat3;
using od_p3pf::Solution;
using od_p3pf::Vec2;
using od_p3pf::Vec3;

constexpr double kPi = 3.141592653589793238462643383279502884;

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 mul(const Mat3& a, const Vec3& x) {
  return {a(0, 0) * x.x + a(0, 1) * x.y + a(0, 2) * x.z,
          a(1, 0) * x.x + a(1, 1) * x.y + a(1, 2) * x.z,
          a(2, 0) * x.x + a(2, 1) * x.y + a(2, 2) * x.z};
}
Vec3 transpose_mul(const Mat3& a, const Vec3& x) {
  return {a(0, 0) * x.x + a(1, 0) * x.y + a(2, 0) * x.z,
          a(0, 1) * x.x + a(1, 1) * x.y + a(2, 1) * x.z,
          a(0, 2) * x.x + a(1, 2) * x.y + a(2, 2) * x.z};
}
double norm(const Vec3& a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double triangle_quality(const std::array<Vec3, 3>& points) {
  const Vec3 a = points[1] - points[0];
  const Vec3 b = points[2] - points[0];
  const Vec3 c = points[2] - points[1];
  const double denominator = norm(a) * norm(a) + norm(b) * norm(b) + norm(c) * norm(c);
  return denominator > 0.0 ? norm(cross(a, b)) / denominator : 0.0;
}

Mat3 random_rotation(std::mt19937_64& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  double w = normal(rng), x = normal(rng), y = normal(rng), z = normal(rng);
  const double inverse = 1.0 / std::sqrt(w * w + x * x + y * y + z * z);
  w *= inverse; x *= inverse; y *= inverse; z *= inverse;
  Mat3 r;
  r(0, 0) = 1.0 - 2.0 * (y * y + z * z);
  r(0, 1) = 2.0 * (x * y - z * w);
  r(0, 2) = 2.0 * (x * z + y * w);
  r(1, 0) = 2.0 * (x * y + z * w);
  r(1, 1) = 1.0 - 2.0 * (x * x + z * z);
  r(1, 2) = 2.0 * (y * z - x * w);
  r(2, 0) = 2.0 * (x * z - y * w);
  r(2, 1) = 2.0 * (y * z + x * w);
  r(2, 2) = 1.0 - 2.0 * (x * x + y * y);
  return r;
}

double rotation_error(const Mat3& a, const Mat3& b) {
  double trace = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) trace += a(i, j) * b(i, j);
  const double cosine = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
  return 180.0 / kPi * std::acos(cosine);
}

struct Scene {
  std::array<Vec3, 3> world;
  std::array<Vec2, 3> pixels;
  std::array<Vec3, 3> camera;
  Mat3 rotation;
  Vec3 translation;
  double focal;
};

Scene random_scene(std::mt19937_64& rng, double pixel_noise, double depth_relative_noise) {
  std::uniform_real_distribution<double> focal_dist(500.0, 1500.0);
  std::uniform_real_distribution<double> x_dist(-2.2, 2.2);
  std::uniform_real_distribution<double> y_dist(-1.6, 1.6);
  std::uniform_real_distribution<double> z_dist(4.0, 10.0);
  std::uniform_real_distribution<double> t_dist(-2.0, 2.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  Scene scene;
  scene.focal = focal_dist(rng);
  scene.rotation = random_rotation(rng);
  scene.translation = {t_dist(rng), t_dist(rng), t_dist(rng)};
  for (int i = 0; i < 3; ++i) {
    scene.camera[i] = {x_dist(rng), y_dist(rng), z_dist(rng)};
    scene.world[i] = transpose_mul(scene.rotation, scene.camera[i] - scene.translation);
    scene.pixels[i] = {
        scene.focal * scene.camera[i].x / scene.camera[i].z + pixel_noise * normal(rng),
        scene.focal * scene.camera[i].y / scene.camera[i].z + pixel_noise * normal(rng)};
  }
  scene.camera[0].z *= 1.0 + depth_relative_noise * normal(rng);
  return scene;
}

Scene conditioned_scene(std::mt19937_64& rng, double epsilon,
                        const std::string& mode, double pixel_noise,
                        double depth_relative_noise) {
  std::uniform_real_distribution<double> focal_dist(700.0, 1200.0);
  std::uniform_real_distribution<double> t_dist(-2.0, 2.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  Scene scene;
  scene.focal = focal_dist(rng);
  scene.rotation = random_rotation(rng);
  scene.translation = {t_dist(rng), t_dist(rng), t_dist(rng)};
  if (mode == "world_triangle") {
    scene.camera = {{{-1.0, 0.0, 6.0}, {0.0, 0.0, 7.0},
                     {1.0, epsilon, 8.0}}};
  } else {
    const Vec2 q1{0.22, 0.06};
    const Vec2 q2{-0.18, 0.15};
    const Vec2 q3{1.35 * q1.x - epsilon * q1.y,
                  1.35 * q1.y + epsilon * q1.x};
    const std::array<Vec2, 3> rays{{q1, q2, q3}};
    const std::array<double, 3> depths{{6.0, 7.5, 8.0}};
    for (int i = 0; i < 3; ++i)
      scene.camera[i] = {depths[i] * rays[i].x, depths[i] * rays[i].y, depths[i]};
  }
  for (int i = 0; i < 3; ++i) {
    scene.world[i] = transpose_mul(scene.rotation, scene.camera[i] - scene.translation);
    scene.pixels[i] = {
        scene.focal * scene.camera[i].x / scene.camera[i].z + pixel_noise * normal(rng),
        scene.focal * scene.camera[i].y / scene.camera[i].z + pixel_noise * normal(rng)};
  }
  scene.camera[0].z *= 1.0 + depth_relative_noise * normal(rng);
  return scene;
}

double median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  if (values.size() % 2) return values[middle];
  const double high = values[middle];
  std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
  return 0.5 * (high + values[middle - 1]);
}

struct TrialSummary {
  int failures = 0;
  int exact_recovery_failures = 0;
  int rotation_over_millidegree = 0;
  double maximum_rotation = 0.0;
  double maximum_translation = 0.0;
  double maximum_focal = 0.0;
  std::vector<double> bad_triangle_quality;
  std::vector<double> rotation;
  std::vector<double> translation;
  std::vector<double> focal;
  std::vector<double> counts;
};

TrialSummary run_trials(int trials, std::uint64_t seed, double pixel_noise,
                        double depth_relative_noise) {
  std::mt19937_64 rng(seed);
  TrialSummary summary;
  for (int trial = 0; trial < trials; ++trial) {
    Scene scene = random_scene(rng, pixel_noise, depth_relative_noise);
    const auto candidates = od_p3pf::solve(scene.world, scene.pixels, scene.camera[0].z);
    summary.counts.push_back(static_cast<double>(candidates.size()));
    if (candidates.empty()) {
      ++summary.failures;
      continue;
    }
    double best_score = std::numeric_limits<double>::infinity();
    const Solution* best = nullptr;
    for (const Solution& candidate : candidates) {
      const double re = rotation_error(candidate.R, scene.rotation);
      const double te = norm(candidate.t - scene.translation);
      const double fe = std::abs(candidate.focal - scene.focal) / scene.focal;
      const double score = re + te + 10.0 * fe;
      if (score < best_score) {
        best_score = score;
        best = &candidate;
      }
    }
    summary.rotation.push_back(rotation_error(best->R, scene.rotation));
    summary.translation.push_back(norm(best->t - scene.translation));
    summary.focal.push_back(std::abs(best->focal - scene.focal) / scene.focal);
    summary.maximum_rotation = std::max(summary.maximum_rotation, summary.rotation.back());
    summary.maximum_translation = std::max(summary.maximum_translation, summary.translation.back());
    summary.maximum_focal = std::max(summary.maximum_focal, summary.focal.back());
    if (summary.rotation.back() > 1e-3) {
      ++summary.rotation_over_millidegree;
      if (pixel_noise == 0.0 && depth_relative_noise == 0.0) {
        const double image_det = scene.pixels[0].x * scene.pixels[2].y -
                                 scene.pixels[0].y * scene.pixels[2].x;
        const double image_scale = std::max(1.0, std::abs(scene.pixels[0].x) +
                                                     std::abs(scene.pixels[0].y));
        std::cerr << "ill_conditioned_exact_case, trial=" << trial
                  << ", rotation_deg=" << summary.rotation.back()
                  << ", triangle_quality=" << triangle_quality(scene.world)
                  << ", normalized_image_det=" << image_det / (image_scale * image_scale)
                  << ", solutions=" << candidates.size() << '\n';
      }
    }
    if (pixel_noise == 0.0 && depth_relative_noise == 0.0 &&
        (summary.rotation.back() > 1e-5 || summary.translation.back() > 1e-7 ||
         summary.focal.back() > 1e-8)) {
      ++summary.exact_recovery_failures;
      summary.bad_triangle_quality.push_back(triangle_quality(scene.world));
    }
  }
  return summary;
}

TrialSummary run_condition_trials(int trials, std::uint64_t seed, double epsilon,
                                  const std::string& mode) {
  std::mt19937_64 rng(seed);
  TrialSummary summary;
  for (int trial = 0; trial < trials; ++trial) {
    Scene scene = conditioned_scene(rng, epsilon, mode, 1.0, 0.005);
    const auto candidates = od_p3pf::solve(scene.world, scene.pixels, scene.camera[0].z);
    summary.counts.push_back(static_cast<double>(candidates.size()));
    if (candidates.empty()) {
      ++summary.failures;
      continue;
    }
    double best_score = std::numeric_limits<double>::infinity();
    const Solution* best = nullptr;
    for (const Solution& candidate : candidates) {
      const double re = rotation_error(candidate.R, scene.rotation);
      const double te = norm(candidate.t - scene.translation);
      const double fe = std::abs(candidate.focal - scene.focal) / scene.focal;
      const double score = re + te + 10.0 * fe;
      if (score < best_score) {
        best_score = score;
        best = &candidate;
      }
    }
    summary.rotation.push_back(rotation_error(best->R, scene.rotation));
    summary.translation.push_back(norm(best->t - scene.translation));
    summary.focal.push_back(std::abs(best->focal - scene.focal) / scene.focal);
    summary.maximum_rotation = std::max(summary.maximum_rotation, summary.rotation.back());
    summary.maximum_translation = std::max(summary.maximum_translation, summary.translation.back());
    summary.maximum_focal = std::max(summary.maximum_focal, summary.focal.back());
    if (summary.rotation.back() > 1e-3) ++summary.rotation_over_millidegree;
  }
  return summary;
}

void print_summary(const std::string& label, int trials, TrialSummary result) {
  const auto quantile = [](std::vector<double> values, double probability) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t index = static_cast<std::size_t>(
        std::floor(probability * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
  };
  const double rotation_p90 = quantile(result.rotation, 0.9);
  const double rotation_p99 = quantile(result.rotation, 0.99);
  const double translation_p90 = quantile(result.translation, 0.9);
  const double focal_p90 = quantile(result.focal, 0.9);
  std::cout << label << ", trials=" << trials << ", failures=" << result.failures
            << ", exact_recovery_failures=" << result.exact_recovery_failures
            << ", rotation_over_1e-3deg=" << result.rotation_over_millidegree
            << ", median_solutions=" << median(std::move(result.counts))
            << ", median_rotation_deg=" << median(std::move(result.rotation))
            << ", rotation_p90_deg=" << rotation_p90
            << ", rotation_p99_deg=" << rotation_p99
            << ", median_translation=" << median(std::move(result.translation))
            << ", translation_p90=" << translation_p90
            << ", median_focal_relative=" << median(std::move(result.focal)) << '\n';
  std::cout << label << "_tail, focal_p90_relative=" << focal_p90 << '\n';
  std::cout << label << "_max, rotation_deg=" << result.maximum_rotation
            << ", translation=" << result.maximum_translation
            << ", focal_relative=" << result.maximum_focal
            << ", bad_triangle_quality_median="
            << median(std::move(result.bad_triangle_quality)) << '\n';
}

void run_sweeps() {
  constexpr int trials = 5000;
  for (double noise : {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}) {
    print_summary("pixel_noise_" + std::to_string(noise), trials,
                  run_trials(trials, 20261001, noise, 0.005));
  }
  for (double noise : {0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05}) {
    print_summary("depth_noise_" + std::to_string(noise), trials,
                  run_trials(trials, 20261002, 1.0, noise));
  }
  constexpr int condition_trials = 2000;
  for (const std::string mode : {"world_triangle", "image_determinant"}) {
    for (double epsilon : {1.0, 0.3, 0.1, 0.03, 0.01, 0.003}) {
      print_summary(mode + "_" + std::to_string(epsilon), condition_trials,
                    run_condition_trials(condition_trials, 20261003, epsilon, mode));
    }
  }
}

bool six_solution_regression() {
  const std::array<Vec2, 3> rays{{{0.016217, -0.524291},
                                  {2.154191, -3.615259},
                                  {-0.554594, -1.396616}}};
  const std::array<double, 3> depth{{5.326722, 1.795670, 4.218296}};
  std::array<Vec3, 3> world{};
  std::array<Vec2, 3> pixels{};
  for (int i = 0; i < 3; ++i) {
    world[i] = {depth[i] * rays[i].x, depth[i] * rays[i].y, depth[i]};
    pixels[i] = {1000.0 * rays[i].x, 1000.0 * rays[i].y};
  }
  const auto solutions = od_p3pf::solve(world, pixels, depth[0]);
  std::cout << "six_solution_witness, returned=" << solutions.size() << ", focals=";
  for (const auto& solution : solutions) std::cout << ' ' << solution.focal;
  std::cout << '\n';
  return solutions.size() == 6;
}

void benchmark_runtime(int repetitions) {
  std::mt19937_64 rng(8181);
  std::vector<Scene> scenes;
  scenes.reserve(256);
  for (int i = 0; i < 256; ++i) scenes.push_back(random_scene(rng, 1.0, 0.005));
  std::size_t checksum = 0;
  for (const auto& scene : scenes)
    checksum += od_p3pf::solve(scene.world, scene.pixels, scene.camera[0].z).size();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repetitions; ++i) {
    const auto& scene = scenes[static_cast<std::size_t>(i) % scenes.size()];
    checksum += od_p3pf::solve(scene.world, scene.pixels, scene.camera[0].z).size();
  }
  const auto stop = std::chrono::steady_clock::now();
  const double microseconds =
      std::chrono::duration<double, std::micro>(stop - start).count() / repetitions;
  std::cout << "runtime, repetitions=" << repetitions << ", mean_us=" << microseconds
            << ", checksum=" << checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  const bool stress = argc > 1 && std::string(argv[1]) == "--stress";
  const bool sweep = argc > 1 && std::string(argv[1]) == "--sweep";
  const int trials = quick ? 200 : (stress ? 100000 : 5000);
  const int repetitions = quick ? 1000 : 100000;
  std::cout << std::setprecision(9);
  const bool witness_ok = six_solution_regression();
  print_summary("noise_free", trials, run_trials(trials, 20260908, 0.0, 0.0));
  print_summary("noise_1px_depth_0.5pct", trials,
                run_trials(trials, 20260909, 1.0, 0.005));
  benchmark_runtime(repetitions);
  if (sweep) run_sweeps();
  return witness_ok ? 0 : 2;
}
