/**
 * @file Tanh.hpp
 * @brief Tanh activation — std::tanh forward, (1-A^2) backward.
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace FlexNN::Activations::detail {

/**
 * @brief Tanh forward: A = tanh(Z) elementwise.
 */
inline Eigen::MatrixXd tanh_forward(const Eigen::MatrixXd& Z) {
  return Z.unaryExpr([](double x) { return std::tanh(x); });
}

/**
 * @brief Tanh backward: dZ = dA * (1 - A^2).
 *
 * Uses A = tanh(Z) from forward, so no second tanh call.
 */
inline Eigen::MatrixXd tanh_backward(const Eigen::MatrixXd& dA,
                                     const Eigen::MatrixXd& /*Z*/,
                                     const Eigen::MatrixXd& A) noexcept {
  return dA.array() * (1.0 - A.array().square());
}

} // namespace FlexNN::Activations::detail
