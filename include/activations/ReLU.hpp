/**
 * @file ReLU.hpp
 * @brief ReLU activation — max(0, z) forward, (z>0) backward.
 */

#pragma once

#include <algorithm> // std::max
#include <Eigen/Dense>

namespace FlexNN::Activations::detail {

/**
 * @brief ReLU forward: A = max(0, Z) elementwise.
 */
inline Eigen::MatrixXd relu_forward(const Eigen::MatrixXd& Z) {
  return Z.unaryExpr([](double x) { return std::max(0.0, x); });
}

/**
 * @brief ReLU backward: dZ = dA * (Z > 0).
 *
 * Note: uses Z, not A, so the dead-neuron boundary (Z==0) is 0 as in
 * the original `lib/Layer.cpp:70` implementation. This matches the
 * subgradient choice for ReLU and avoids a branch on A.
 */
inline Eigen::MatrixXd relu_backward(const Eigen::MatrixXd& dA,
                                     const Eigen::MatrixXd& Z,
                                     const Eigen::MatrixXd& /*A*/) noexcept {
  return dA.array() * (Z.array() > 0.0).cast<double>();
}

} // namespace FlexNN::Activations::detail
