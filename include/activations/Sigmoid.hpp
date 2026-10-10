/**
 * @file Sigmoid.hpp
 * @brief Sigmoid activation — 1/(1+exp(-Z)) with clamping.
 *
 * Clamping Z to [-15,15] avoids exp overflow and keeps host double
 * stable while the runtime may use a LUT. Matches LLD_FLEXNN §6.5.
 */

#pragma once

#include <algorithm> // std::max / std::min
#include <Eigen/Dense>
#include <cmath>

namespace FlexNN::Activations::detail {

/**
 * @brief Sigmoid forward: A = 1/(1+exp(-clamp(Z))) elementwise.
 *
 * Clamps Z to [-15,15] before exp to avoid overflow (exp(15) ~ 3.2M).
 */
inline Eigen::MatrixXd sigmoid_forward(const Eigen::MatrixXd& Z) {
  return Z.unaryExpr([](double x) {
    double z = std::max(-15.0, std::min(15.0, x));
    return 1.0 / (1.0 + std::exp(-z));
  });
}

/**
 * @brief Sigmoid backward: dZ = dA * A * (1 - A).
 *
 * Uses A (already computed) to avoid recomputing exp — matches the
 * standard `d sigmoid / dz = s * (1-s)` form.
 */
inline Eigen::MatrixXd sigmoid_backward(const Eigen::MatrixXd& dA,
                                        const Eigen::MatrixXd& /*Z*/,
                                        const Eigen::MatrixXd& A) noexcept {
  return dA.array() * A.array() * (1.0 - A.array());
}

} // namespace FlexNN::Activations::detail
