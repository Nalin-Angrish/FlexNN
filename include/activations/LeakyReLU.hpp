/**
 * @file LeakyReLU.hpp
 * @brief Leaky ReLU with fixed alpha=0.01 (v0.1).
 *
 * Why fixed: `model.bin` encodes activation as a single uint8_t (no per-layer
 * float for alpha in v0.1). A future per-layer alpha would be stored in the
 * `aux` blob (one float32) and this header would gain an overload.
 */

#pragma once

#include <Eigen/Dense>

namespace FlexNN::Activations::detail {

// Fixed slope for negative side — matches LLD_FLEXNN §4.
inline constexpr double kLeakyAlpha = 0.01;

/**
 * @brief LeakyReLU forward: A = Z>0 ? Z : 0.01*Z.
 */
inline Eigen::MatrixXd leaky_relu_forward(const Eigen::MatrixXd& Z) {
  return Z.unaryExpr([](double x) { return x > 0.0 ? x : kLeakyAlpha * x; });
}

/**
 * @brief LeakyReLU backward: dZ = dA * (Z>0 ? 1 : 0.01).
 */
inline Eigen::MatrixXd leaky_relu_backward(const Eigen::MatrixXd& dA,
                                           const Eigen::MatrixXd& Z,
                                           const Eigen::MatrixXd& /*A*/) noexcept {
  return dA.array() * (Z.array() > 0.0).cast<double>() +
         dA.array() * (Z.array() <= 0.0).cast<double>() * kLeakyAlpha;
  // Equivalent to: (Z>0 ? 1 : 0.01) elementwise, but keeps Eigen vectorization.
}

} // namespace FlexNN::Activations::detail
