/**
 * @file LeakyReLU.hpp
 * @brief Leaky ReLU with configurable alpha (default 0.01).
 *
 * `alpha` is the slope for `z <= 0`. Historic fixed value 0.01 (v0.1) is kept
 * as `kDefaultLeakyAlpha` in `ActivationParameters.hpp`; this header now takes
 * `alpha` as a parameter so layers can store per-layer `ActivationParameters`.
 */

#pragma once

#include <Eigen/Dense>

namespace FlexNN::Activations::detail {

// Default kept for backward compat; prefer Activations::kDefaultLeakyAlpha.
inline constexpr double kLeakyAlpha = 0.01;

/**
 * @brief LeakyReLU forward: A = Z>0 ? Z : alpha*Z.
 * @param Z Pre-activation
 * @param alpha Slope for z<=0 (0 < alpha < 1, default 0.01)
 */
inline Eigen::MatrixXd leaky_relu_forward(const Eigen::MatrixXd& Z,
                                          double alpha = kLeakyAlpha) {
  return Z.unaryExpr([alpha](double x) { return x > 0.0 ? x : alpha * x; });
}

/**
 * @brief LeakyReLU backward: dZ = dA * (Z>0 ? 1 : alpha).
 * @param dA Upstream gradient
 * @param Z Pre-activation from forward
 * @param alpha Same alpha used in forward
 */
inline Eigen::MatrixXd leaky_relu_backward(const Eigen::MatrixXd& dA,
                                           const Eigen::MatrixXd& Z,
                                           const Eigen::MatrixXd& /*A*/,
                                           double alpha = kLeakyAlpha) noexcept {
  return dA.array() * (Z.array() > 0.0).cast<double>() +
         dA.array() * (Z.array() <= 0.0).cast<double>() * alpha;
  // Equivalent to: (Z>0 ? 1 : alpha) elementwise, keeps Eigen vectorization.
}

} // namespace FlexNN::Activations::detail
