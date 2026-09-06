/**
 * @file Softmax.hpp
 * @brief Stable column-wise softmax.
 *
 * Forward is `exp(Z - max(col)) / sum(exp(...))` per column (batch).
 * Backward for hidden Softmax is `J * upstream` where J = diag(s) - s s^T.
 * Last-layer Softmax + cross-entropy is fused as `A - Y` in
 * `NeuralNetwork::backward()` and never calls the hidden path (see LLD §4.3).
 */

#pragma once

#include <Eigen/Dense>

namespace FlexNN::Activations::detail {

/**
 * @brief Stable column-wise softmax forward.
 *
 * @param Z Pre-activation [out × batch] (each column is a sample)
 * @return Softmax output, same shape, columns sum to 1
 */
inline Eigen::MatrixXd softmax_forward(const Eigen::MatrixXd& Z) {
  Eigen::MatrixXd A(Z.rows(), Z.cols());
  for (Eigen::Index c = 0; c < Z.cols(); ++c) {
    Eigen::VectorXd col = Z.col(c);
    double m = col.maxCoeff();
    Eigen::VectorXd e = (col.array() - m).exp();
    A.col(c) = e / e.sum();
  }
  return A;
}

/**
 * @brief Softmax hidden backward per column: `dZ = J * upstream`.
 *
 * For each column, `s` is the softmax output for that column and `up` is
 * the upstream gradient column (`W_next^T * dZ_next` for that sample).
 * The Jacobian is `J = diag(s) - s * s^T`, so `J*up = s ⊙ (up - s^T up)`.
 * Implemented without forming J explicitly for O(n) per column.
 *
 * @param s Softmax output per column [out × batch]
 * @param upstream Upstream gradient [out × batch]
 * @return Gradient w.r.t. Z, same shape
 */
inline Eigen::MatrixXd softmax_backward_hidden(const Eigen::MatrixXd& s,
                                              const Eigen::MatrixXd& upstream) noexcept {
  Eigen::MatrixXd dZ(s.rows(), s.cols());
  for (Eigen::Index c = 0; c < s.cols(); ++c) {
    Eigen::VectorXd sv = s.col(c);
    Eigen::VectorXd up = upstream.col(c);
    double dot = sv.dot(up); // s^T * up (scalar per column)
    // J*up = diag(s)*up - s*(s^T*up) = s ⊙ up - s*dot = s ⊙ (up - dot)
    dZ.col(c) = sv.array() * up.array() - sv.array() * dot;
  }
  return dZ;
}

} // namespace FlexNN::Activations::detail
