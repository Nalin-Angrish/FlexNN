/**
 * @file Utility.h
 * @brief Utility functions for FlexNN — CSV, one-hot, and deterministic splits.
 *
 * Adds `setRandomSeed`/`getRandomSeed` so `splitXY` is deterministic when a
 * seed is set, while keeping the old `splitXY(X,Y,props)` signature for
 * backward compat (uses the global seed). See LLD_FLEXNN §8.
 */

#ifndef FlexNN_UTILITY_H
#define FlexNN_UTILITY_H

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace FlexNN {

/**
 * @brief One-hot encodes a vector of class labels.
 *
 * @param Y Input vector of class labels (size = batch)
 * @param num_classes Number of unique classes
 * @return Matrix [num_classes × batch] where each column is one-hot
 */
Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd &Y, int num_classes);

/**
 * @brief Reads a CSV where first column is label, rest are features.
 *
 * @param filename Path to CSV
 * @param X Output features [samples × features]
 * @param Y Output labels [samples]
 */
void readCSV_XY(const std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y);

/**
 * @brief Set the global RNG seed for `splitXY` (deterministic splits).
 *
 * Thread-safe (atomic). The `thread_local` engine is reseeded on next
 * `splitXY` call without explicit seed. For per-call determinism, use
 * `splitXY(X,Y,props,seed)` instead.
 *
 * @param seed New seed
 */
void setRandomSeed(uint32_t seed);

/**
 * @brief Get the current global RNG seed.
 *
 * @return Current seed (initially from `random_device` once)
 */
uint32_t getRandomSeed() noexcept;

/**
 * @brief Splits dataset into multiple sets by proportions (uses global seed).
 *
 * Shuffles indices via a `thread_local mt19937` seeded from the global
 * `getRandomSeed()` (or `random_device` on first call). For reproducibility,
 * call `setRandomSeed(42)` before `splitXY`, or use the overload with
 * explicit `seed`.
 *
 * @param X Features [samples × features]
 * @param Y Labels [samples]
 * @param proportions Split fractions (sum ≤1, last is remainder)
 * @return Vector of (X_split, Y_split) pairs
 */
std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splitXY(
    const Eigen::MatrixXd &X, const Eigen::VectorXd &Y,
    const std::vector<double> &proportions);

/**
 * @brief Splits dataset with explicit seed (deterministic, no global state).
 *
 * Uses a local `mt19937(seed)` so the global seed is unchanged. Prefer this
 * in tests for hermetic determinism.
 *
 * @param X Features [samples × features]
 * @param Y Labels [samples]
 * @param proportions Split fractions
 * @param seed Explicit seed
 * @return Vector of splits
 */
std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splitXY(
    const Eigen::MatrixXd &X, const Eigen::VectorXd &Y,
    const std::vector<double> &proportions, uint32_t seed);

} // namespace FlexNN

#endif // FlexNN_UTILITY_H
