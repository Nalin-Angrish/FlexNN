/**
 * @file Utility.cpp
 * @brief Implements FlexNN utilities with deterministic seeding.
 *
 * Why this file: `splitXY` was nondeterministic (`random_device` per call)
 * which broke test hermeticity. Now `setRandomSeed` + `thread_local mt19937`
 * makes it deterministic when needed, while keeping the old signature for
 * backward compat (see LLD_FLEXNN §8).
 */

#include "Utility.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace FlexNN {

// Global seed — initially random_device once, then setRandomSeed overwrites.
// Use atomic for thread-safe set/get.
static std::atomic<uint32_t> g_seed([] {
  std::random_device rd;
  return rd();
}());

// Thread-local engine seeded from global seed on first use.
static thread_local std::mt19937 g_rng(g_seed.load());

void setRandomSeed(uint32_t seed) {
  g_seed.store(seed);
  g_rng.seed(seed);
}

uint32_t getRandomSeed() noexcept { return g_seed.load(); }

Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd &Y, int num_classes) {
  Eigen::MatrixXd Y_onehot = Eigen::MatrixXd::Zero(num_classes, Y.size());
  for (Eigen::Index i = 0; i < Y.size(); ++i) {
    int label = static_cast<int>(Y(i));
    if (label >= 0 && label < num_classes) Y_onehot(label, i) = 1.0;
  }
  return Y_onehot;
}

void readCSV_XY(const std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y) {
  std::ifstream file(filename);
  std::vector<std::vector<double>> data;
  std::string line;
  size_t cols = 0;

  if (std::getline(file, line)) {
    // Skip header
  }
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell));
    }
    if (cols == 0) cols = row.size();
    data.push_back(std::move(row));
  }

  size_t nRows = data.size();
  size_t nCols = cols;
  X.resize(nRows, nCols - 1);
  Y.resize(nRows);
  for (size_t i = 0; i < nRows; ++i) {
    Y(i) = data[i][0];
    for (size_t j = 1; j < nCols; ++j) {
      X(i, j - 1) = data[i][j];
    }
  }
}

std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splitXY(
    const Eigen::MatrixXd &X, const Eigen::VectorXd &Y,
    const std::vector<double> &proportions) {
  // Use thread-local engine seeded from global seed
  // Reseed if global seed changed since last call (detects setRandomSeed)
  static thread_local uint32_t last_seed = g_seed.load();
  uint32_t cur = g_seed.load();
  if (cur != last_seed) {
    g_rng.seed(cur);
    last_seed = cur;
  }

  size_t nRows = static_cast<size_t>(X.rows());
  std::vector<size_t> indices(nRows);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), g_rng);

  std::vector<size_t> sizes;
  sizes.reserve(proportions.size());
  size_t total = 0;
  for (double p : proportions) {
    size_t sz = static_cast<size_t>(p * nRows);
    sizes.push_back(sz);
    total += sz;
  }
  if (!sizes.empty()) sizes.back() += nRows - total;

  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splits;
  splits.reserve(sizes.size());
  size_t start = 0;
  for (size_t k = 0; k < sizes.size(); ++k) {
    size_t sz = sizes[k];
    Eigen::MatrixXd X_split(sz, X.cols());
    Eigen::VectorXd Y_split(sz);
    for (size_t i = 0; i < sz; ++i) {
      X_split.row(i) = X.row(indices[start + i]);
      Y_split(i) = Y(indices[start + i]);
    }
    splits.emplace_back(std::move(X_split), std::move(Y_split));
    start += sz;
  }
  return splits;
}

std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splitXY(
    const Eigen::MatrixXd &X, const Eigen::VectorXd &Y,
    const std::vector<double> &proportions, uint32_t seed) {
  // Explicit seed — local engine, no global state change
  std::mt19937 local_rng(seed);
  size_t nRows = static_cast<size_t>(X.rows());
  std::vector<size_t> indices(nRows);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), local_rng);

  std::vector<size_t> sizes;
  sizes.reserve(proportions.size());
  size_t total = 0;
  for (double p : proportions) {
    size_t sz = static_cast<size_t>(p * nRows);
    sizes.push_back(sz);
    total += sz;
  }
  if (!sizes.empty()) sizes.back() += nRows - total;

  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splits;
  splits.reserve(sizes.size());
  size_t start = 0;
  for (size_t k = 0; k < sizes.size(); ++k) {
    size_t sz = sizes[k];
    Eigen::MatrixXd X_split(sz, X.cols());
    Eigen::VectorXd Y_split(sz);
    for (size_t i = 0; i < sz; ++i) {
      X_split.row(i) = X.row(indices[start + i]);
      Y_split(i) = Y(indices[start + i]);
    }
    splits.emplace_back(std::move(X_split), std::move(Y_split));
    start += sz;
  }
  return splits;
}

} // namespace FlexNN
