/**
 * @file None.hpp
 * @brief Linear (no) activation — identity forward/backward.
 *
 * Separated into its own header so `tests/activation_test.cpp` can include
 * and test this activation in isolation without pulling all activations.
 */

#pragma once

#include <Eigen/Dense>

namespace FlexNN::Activations::detail {

/**
 * @brief Identity forward: A = Z.
 */
inline Eigen::MatrixXd none_forward(const Eigen::MatrixXd& Z) {
  return Z;
}

/**
 * @brief Identity backward: dZ = dA.
 */
inline Eigen::MatrixXd none_backward(const Eigen::MatrixXd& dA,
                                     const Eigen::MatrixXd& /*Z*/,
                                     const Eigen::MatrixXd& /*A*/) noexcept {
  return dA;
}

} // namespace FlexNN::Activations::detail
