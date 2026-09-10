#include "od_p3pf/od_p3pf.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using od_p3pf::Mat3;
using od_p3pf::Solution;
using od_p3pf::Vec2;
using od_p3pf::Vec3;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Match {
  Vec3 world;
  Vec2 pixel;
  double query_z = 0.0;
  double metric = 0.0;
};

struct Frame {
  int id = 0;
  double timestamp_ref = 0.0;
  double timestamp_query = 0.0;
  double focal_gt = 0.0;
  Mat3 rotation_gt;
  Vec3 translation_gt;
  std::vector<Match> matches;
};

struct Estimate {
  bool valid = false;
  Solution solution;
  int anchor = -1;
  int inliers = 0;
  double msac = std::numeric_limits<double>::infinity();
  int hypotheses = 0;
  double milliseconds = 0.0;
};

Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 mul(const Mat3& a, const Vec3& x) {
  return {a(0, 0) * x.x + a(0, 1) * x.y + a(0, 2) * x.z,
          a(1, 0) * x.x + a(1, 1) * x.y + a(1, 2) * x.z,
          a(2, 0) * x.x + a(2, 1) * x.y + a(2, 2) * x.z};
}
double norm(const Vec3& x) {
  return std::sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
}

double reprojection_error(const Solution& solution, const Match& match) {
  const Vec3 camera = mul(solution.R, match.world) + solution.t;
  if (!(camera.z > 1e-9) || !(solution.focal > 0.0) ||
      !std::isfinite(solution.focal))
    return std::numeric_limits<double>::infinity();
  const double dx = solution.focal * camera.x / camera.z - match.pixel.x;
  const double dy = solution.focal * camera.y / camera.z - match.pixel.y;
  return std::sqrt(dx * dx + dy * dy);
}

double gt_reprojection_error(const Frame& frame, const Match& match) {
  Solution gt;
  gt.R = frame.rotation_gt;
  gt.t = frame.translation_gt;
  gt.focal = frame.focal_gt;
  return reprojection_error(gt, match);
}

double gt_depth_error(const Frame& frame, const Match& match) {
  const Vec3 camera = mul(frame.rotation_gt, match.world) + frame.translation_gt;
  return std::abs(camera.z - match.query_z);
}

double rotation_error(const Mat3& estimate, const Mat3& truth) {
  double trace = 0.0;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      trace += estimate(row, col) * truth(row, col);
  return 180.0 / kPi *
         std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
}

std::vector<Frame> load_frames(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open input: " + path);
  std::vector<Frame> frames;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream header(line);
    std::string tag;
    header >> tag;
    if (tag != "FRAME")
      throw std::runtime_error("Expected FRAME, got: " + line);
    Frame frame;
    int count = 0;
    header >> frame.id >> count >> frame.timestamp_ref >> frame.timestamp_query;

    if (!std::getline(input, line)) throw std::runtime_error("Missing INTRINSICS");
    std::istringstream intrinsics(line);
    double fy = 0.0, cx = 0.0, cy = 0.0;
    intrinsics >> tag >> frame.focal_gt >> fy >> cx >> cy;
    if (tag != "INTRINSICS") throw std::runtime_error("Malformed INTRINSICS");

    if (!std::getline(input, line)) throw std::runtime_error("Missing POSE");
    std::istringstream pose(line);
    pose >> tag;
    if (tag != "POSE") throw std::runtime_error("Malformed POSE");
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) pose >> frame.rotation_gt(row, col);
      if (row == 0) pose >> frame.translation_gt.x;
      if (row == 1) pose >> frame.translation_gt.y;
      if (row == 2) pose >> frame.translation_gt.z;
    }

    frame.matches.reserve(count);
    for (int i = 0; i < count; ++i) {
      if (!std::getline(input, line)) throw std::runtime_error("Truncated matches");
      std::istringstream values(line);
      Match match;
      values >> match.world.x >> match.world.y >> match.world.z >>
          match.pixel.x >> match.pixel.y >> match.query_z >> match.metric;
      if (!values) throw std::runtime_error("Malformed match row: " + line);
      frame.matches.push_back(match);
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

int verified_anchor(const Frame& frame, double pixel_threshold,
                    double absolute_depth_threshold,
                    double relative_depth_threshold) {
  for (std::size_t i = 0; i < frame.matches.size(); ++i) {
    const Match& match = frame.matches[i];
    const double depth_threshold =
        std::max(absolute_depth_threshold, relative_depth_threshold * match.query_z);
    if (gt_reprojection_error(frame, match) < pixel_threshold &&
        gt_depth_error(frame, match) < depth_threshold)
      return static_cast<int>(i);
  }
  return -1;
}

bool valid_query_depth(const Match& match) {
  return std::isfinite(match.query_z) && match.query_z > 0.0;
}

bool is_verified_anchor(const Frame& frame, int index, double pixel_threshold,
                        double absolute_depth_threshold,
                        double relative_depth_threshold) {
  const Match& match = frame.matches[index];
  const double depth_threshold =
      std::max(absolute_depth_threshold, relative_depth_threshold * match.query_z);
  return valid_query_depth(match) &&
         gt_reprojection_error(frame, match) < pixel_threshold &&
         gt_depth_error(frame, match) < depth_threshold;
}

Estimate run_fixed_anchor_ransac(const Frame& frame, int anchor, int iterations,
                                 double pixel_threshold, std::mt19937_64& rng) {
  Estimate result;
  if (anchor < 0 || frame.matches.size() < 3) return result;
  std::uniform_int_distribution<int> index_dist(
      0, static_cast<int>(frame.matches.size()) - 1);
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    int second = index_dist(rng);
    int third = index_dist(rng);
    if (second == anchor || third == anchor || second == third) {
      --iteration;
      continue;
    }
    const std::array<int, 3> ids{{anchor, second, third}};
    std::array<Vec3, 3> world;
    std::array<Vec2, 3> pixels;
    for (int i = 0; i < 3; ++i) {
      world[i] = frame.matches[ids[i]].world;
      pixels[i] = frame.matches[ids[i]].pixel;
    }
    const auto solutions =
        od_p3pf::solve(world, pixels, frame.matches[anchor].query_z);
    result.hypotheses += static_cast<int>(solutions.size());
    for (const Solution& solution : solutions) {
      int inliers = 0;
      double msac = 0.0;
      const double threshold_squared = pixel_threshold * pixel_threshold;
      for (const Match& match : frame.matches) {
        const double error = reprojection_error(solution, match);
        if (error < pixel_threshold) ++inliers;
        msac += std::min(error * error, threshold_squared);
      }
      if (!result.valid || inliers > result.inliers ||
          (inliers == result.inliers && msac < result.msac)) {
        result.valid = true;
        result.solution = solution;
        result.anchor = anchor;
        result.inliers = inliers;
        result.msac = msac;
      }
    }
  }
  result.milliseconds = 1e3 * std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
  return result;
}

Estimate run_candidate_anchor_ransac(
    const Frame& frame, const std::vector<int>& anchors,
    const std::vector<double>& anchor_weights, int iterations,
    double pixel_threshold, std::mt19937_64& rng) {
  Estimate result;
  if (anchors.empty() || frame.matches.size() < 3) return result;
  std::discrete_distribution<int> anchor_dist(anchor_weights.begin(),
                                               anchor_weights.end());
  std::uniform_int_distribution<int> index_dist(
      0, static_cast<int>(frame.matches.size()) - 1);
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int anchor = anchors[anchor_dist(rng)];
    int second = index_dist(rng);
    int third = index_dist(rng);
    if (second == anchor || third == anchor || second == third) {
      --iteration;
      continue;
    }
    const std::array<int, 3> ids{{anchor, second, third}};
    std::array<Vec3, 3> world;
    std::array<Vec2, 3> pixels;
    for (int i = 0; i < 3; ++i) {
      world[i] = frame.matches[ids[i]].world;
      pixels[i] = frame.matches[ids[i]].pixel;
    }
    const auto solutions =
        od_p3pf::solve(world, pixels, frame.matches[anchor].query_z);
    result.hypotheses += static_cast<int>(solutions.size());
    for (const Solution& solution : solutions) {
      int inliers = 0;
      double msac = 0.0;
      const double threshold_squared = pixel_threshold * pixel_threshold;
      for (const Match& match : frame.matches) {
        const double error = reprojection_error(solution, match);
        if (error < pixel_threshold) ++inliers;
        msac += std::min(error * error, threshold_squared);
      }
      if (!result.valid || inliers > result.inliers ||
          (inliers == result.inliers && msac < result.msac)) {
        result.valid = true;
        result.solution = solution;
        result.anchor = anchor;
        result.inliers = inliers;
        result.msac = msac;
      }
    }
  }
  result.milliseconds = 1e3 * std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
  return result;
}

double median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  if (values.size() % 2 == 1) return values[middle];
  const double high = values[middle];
  std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
  return 0.5 * (high + values[middle - 1]);
}

void append_result(std::ofstream& output, const Frame& frame,
                   const std::string& anchor_mode, int anchor,
                   const Estimate& estimate) {
  const Match& anchor_match = frame.matches[anchor];
  const double anchor_reprojection = gt_reprojection_error(frame, anchor_match);
  const double anchor_depth = gt_depth_error(frame, anchor_match);
  output << frame.id << ',' << anchor_mode << ',' << frame.matches.size() << ','
         << anchor << ',' << anchor_match.query_z << ',' << anchor_reprojection << ','
         << anchor_depth << ','
         << estimate.valid << ',' << estimate.inliers << ',' << estimate.hypotheses
         << ',' << estimate.milliseconds;
  if (estimate.valid) {
    output << ',' << rotation_error(estimate.solution.R, frame.rotation_gt) << ','
           << norm(estimate.solution.t - frame.translation_gt) << ','
           << std::abs(estimate.solution.focal - frame.focal_gt) / frame.focal_gt;
  } else {
    output << ",nan,nan,nan";
  }
  output << '\n';
}

std::vector<int> top_depth_anchors(const Frame& frame, int requested) {
  std::vector<int> anchors;
  for (std::size_t i = 0; i < frame.matches.size() &&
                          static_cast<int>(anchors.size()) < requested;
       ++i) {
    if (valid_query_depth(frame.matches[i])) anchors.push_back(static_cast<int>(i));
  }
  return anchors;
}

std::vector<double> descriptor_weights(const Frame& frame,
                                       const std::vector<int>& anchors,
                                       bool weighted) {
  std::vector<double> weights(anchors.size(), 1.0);
  if (!weighted || anchors.size() <= 1) return weights;
  const double first = frame.matches[anchors.front()].metric;
  const double last = frame.matches[anchors.back()].metric;
  const double spread = last - first;
  if (!(spread > 1e-12)) return weights;
  const double slope = std::log(static_cast<double>(anchors.size())) / spread;
  for (std::size_t i = 0; i < anchors.size(); ++i) {
    weights[i] = std::exp(-slope * (frame.matches[anchors[i]].metric - first));
  }
  return weights;
}

void append_candidate_result(
    std::ofstream& output, const Frame& frame, const std::string& sampling,
    int requested_k, const std::vector<int>& anchors,
    int correct_candidates, double alpha, int visual_inliers, int iterations,
    double pixel_threshold, double absolute_depth_threshold,
    double relative_depth_threshold, const Estimate& estimate) {
  const double visual_inlier_rate =
      static_cast<double>(visual_inliers) /
      std::max<std::size_t>(1, frame.matches.size());
  const double pair_success =
      visual_inliers >= 3 && frame.matches.size() >= 3
          ? static_cast<double>((visual_inliers - 1) * (visual_inliers - 2)) /
                static_cast<double>((frame.matches.size() - 1) *
                                    (frame.matches.size() - 2))
          : 0.0;
  const double trial_success = alpha * pair_success;
  const double predicted_success =
      1.0 - std::pow(1.0 - trial_success, iterations);
  const bool selected_correct =
      estimate.anchor >= 0 &&
      is_verified_anchor(frame, estimate.anchor, pixel_threshold,
                         absolute_depth_threshold, relative_depth_threshold);
  output << frame.id << ',' << sampling << ',' << requested_k << ','
         << frame.matches.size() << ',' << anchors.size() << ','
         << correct_candidates << ',' << (correct_candidates > 0) << ','
         << alpha << ',' << visual_inlier_rate << ',' << predicted_success << ','
         << estimate.valid << ',' << estimate.inliers << ',' << estimate.hypotheses
         << ',' << estimate.milliseconds << ',' << estimate.anchor << ','
         << selected_correct;
  if (estimate.valid) {
    output << ',' << rotation_error(estimate.solution.R, frame.rotation_gt) << ','
           << norm(estimate.solution.t - frame.translation_gt) << ','
           << std::abs(estimate.solution.focal - frame.focal_gt) / frame.focal_gt;
  } else {
    output << ",nan,nan,nan";
  }
  output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: od_p3pf_tum <export.txt> <results.csv> [iterations] "
                 "[seed] [candidate]\n";
    return 2;
  }
  const int iterations = argc > 3 ? std::stoi(argv[3]) : 1000;
  const std::uint64_t seed = argc > 4 ? std::stoull(argv[4]) : 20260908ULL;
  constexpr double pixel_threshold = 4.0;
  constexpr double anchor_depth_absolute = 0.05;
  constexpr double anchor_depth_relative = 0.03;

  try {
    const std::vector<Frame> frames = load_frames(argv[1]);
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("Cannot open output file");
    output << std::setprecision(12);
    const bool candidate_mode = argc > 5 && std::string(argv[5]) == "candidate";
    if (candidate_mode) {
      output << "frame,sampling,K,matches,valid_depth_candidates,correct_candidates,"
                "correct_anchor_contained,alpha,visual_inlier_rate,"
                "predicted_sampling_success,valid,inliers,hypotheses,runtime_ms,"
                "selected_anchor,selected_anchor_correct,rotation_deg,translation_m,"
                "focal_relative\n";
      const std::array<int, 5> pool_sizes{{1, 3, 5, 10, 20}};
      int completed = 0;
      for (const Frame& frame : frames) {
        int visual_inliers = 0;
        for (const Match& match : frame.matches)
          if (gt_reprojection_error(frame, match) < pixel_threshold)
            ++visual_inliers;
        for (const int requested_k : pool_sizes) {
          const std::vector<int> anchors = top_depth_anchors(frame, requested_k);
          for (const bool weighted : {false, true}) {
            const std::vector<double> weights =
                descriptor_weights(frame, anchors, weighted);
            double total_weight = 0.0;
            double correct_weight = 0.0;
            int correct_candidates = 0;
            for (std::size_t i = 0; i < anchors.size(); ++i) {
              total_weight += weights[i];
              if (is_verified_anchor(frame, anchors[i], pixel_threshold,
                                     anchor_depth_absolute,
                                     anchor_depth_relative)) {
                ++correct_candidates;
                correct_weight += weights[i];
              }
            }
            const double alpha =
                total_weight > 0.0 ? correct_weight / total_weight : 0.0;
            const std::uint64_t configuration_seed =
                seed ^ (static_cast<std::uint64_t>(frame.id + 1) *
                        0x9e3779b97f4a7c15ULL) ^
                (static_cast<std::uint64_t>(requested_k) *
                 0xbf58476d1ce4e5b9ULL) ^
                ((weighted && requested_k > 1) ? 0x94d049bb133111ebULL : 0ULL);
            std::mt19937_64 configuration_rng(configuration_seed);
            const Estimate estimate = run_candidate_anchor_ransac(
                frame, anchors, weights, iterations, pixel_threshold,
                configuration_rng);
            append_candidate_result(
                output, frame, weighted ? "weighted" : "uniform", requested_k,
                anchors, correct_candidates, alpha, visual_inliers,
                iterations, pixel_threshold, anchor_depth_absolute,
                anchor_depth_relative, estimate);
          }
        }
        ++completed;
        if (completed % 10 == 0 || completed == static_cast<int>(frames.size()))
          std::cout << "Candidate-anchor: " << completed << '/' << frames.size()
                    << " frames\n";
      }
      return 0;
    }
    output << "frame,anchor_mode,matches,anchor_index,anchor_query_z_m,"
              "anchor_gt_reproj_px,anchor_gt_depth_m,valid,inliers,hypotheses,runtime_ms,"
              "rotation_deg,translation_m,focal_relative\n";

    std::mt19937_64 rng(seed);
    std::vector<double> gt_reprojection_medians;
    std::vector<double> gt_depth_medians;
    int total_gt_image_inliers = 0;
    int total_matches = 0;
    int strongest_verified = 0;
    int oracle_frames = 0;
    for (const Frame& frame : frames) {
      std::vector<double> frame_reprojection;
      std::vector<double> frame_depth;
      for (const Match& match : frame.matches) {
        const double reprojection = gt_reprojection_error(frame, match);
        const double depth = gt_depth_error(frame, match);
        if (reprojection < pixel_threshold) {
          ++total_gt_image_inliers;
          frame_reprojection.push_back(reprojection);
          frame_depth.push_back(depth);
        }
        ++total_matches;
      }
      gt_reprojection_medians.push_back(median(frame_reprojection));
      gt_depth_medians.push_back(median(frame_depth));

      const int oracle = verified_anchor(frame, pixel_threshold,
                                         anchor_depth_absolute,
                                         anchor_depth_relative);
      if (oracle >= 0) ++oracle_frames;
      if (oracle == 0) ++strongest_verified;

      const Estimate strongest = run_fixed_anchor_ransac(
          frame, 0, iterations, pixel_threshold, rng);
      append_result(output, frame, "strongest", 0, strongest);
      if (oracle >= 0) {
        const Estimate verified = run_fixed_anchor_ransac(
            frame, oracle, iterations, pixel_threshold, rng);
        append_result(output, frame, "verified", oracle, verified);
      }
    }

    std::cout << std::fixed << std::setprecision(4)
              << "frames=" << frames.size() << " matches=" << total_matches
              << " gt_reprojection_inlier_rate="
              << static_cast<double>(total_gt_image_inliers) /
                     std::max(1, total_matches)
              << " median_frame_gt_reprojection_px="
              << median(gt_reprojection_medians)
              << " median_frame_gt_depth_error_m=" << median(gt_depth_medians)
              << " verified_anchor_frames=" << oracle_frames
              << " strongest_is_verified=" << strongest_verified << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
