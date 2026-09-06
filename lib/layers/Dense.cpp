/**
 * @file Dense.cpp
 * @brief Implements FlexNN::Layers::Dense forward/backward/update.
 *
 * Forward uses `Activations::detail::forward` for the activation step
 * so per-activation logic is tested once in `tests/activation_test.cpp`.
 * Backward uses `Activations::detail::backward` which for Softmax hidden
 * applies `J = diag(s)-s s^T` via `softmax_backward_hidden`.
 */

#include "layers/Dense.hpp"

#include <cassert>

#include "activations/detail.hpp"

namespace FlexNN::Layers {

Dense::Dense(int in, int out, Activations::Activation act)
    : params_{in, out}, act_(act) {
  // Same initialization as legacy `include/Layer.h:55` (Random * 0.5)
  // to keep v0.1 training behaviour comparable.
  W_ = Eigen::MatrixXd::Random(out, in) * 0.5;
  b_ = Eigen::VectorXd::Random(out) * 0.5;
}

Dense::Dense(DenseParams p, Activations::Activation act)
    : Dense(p.inputSize, p.outputSize, act) {}

Dense::Dense(int in, int out, const std::string& actStr)
    : Dense(in, out, Activations::Activation::ReLU) {
  Activations::Activation parsed;
  if (Activations::try_parse(actStr, parsed)) {
    act_ = parsed;
  } else {
    act_ = Activations::Activation::None;
  }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Dense::forward(
    const Eigen::MatrixXd& input) const {
  // Linear: Z = W*X + b (b broadcast column-wise)
  Eigen::MatrixXd Z = (W_ * input).colwise() + b_;
  // Activation: A = f(Z)
  Eigen::MatrixXd A = Activations::detail::forward(act_, Z);
  return {Z, A};
}

Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& nextW,
                                const Eigen::MatrixXd& nextdZ,
                                const Eigen::MatrixXd& currZ) const {
  // Defensive: nextW should match currZ dims for hidden layers.
  // Last-layer is never called (NeuralNetwork::backward fuses Softmax).
  assert(nextW.cols() == currZ.rows() && "nextW cols must equal currZ rows");
  assert(nextW.rows() == nextdZ.rows() && "nextW rows must equal nextdZ rows");
  // Upstream gradient: dA = W_next^T * dZ_next
  Eigen::MatrixXd dA = nextW.transpose() * nextdZ;
  return backward(dA, currZ);
}

Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& upstream,
                                const Eigen::MatrixXd& currZ) const {
  // Need A for Sigmoid/Tanh to avoid recompute; recompute via forward.
  Eigen::MatrixXd A = Activations::detail::forward(act_, currZ);
  return Activations::detail::backward(act_, upstream, currZ, A);
}

Eigen::MatrixXd Dense::propagate(const Eigen::MatrixXd& dZ) const {
  return W_.transpose() * dZ;
}

void Dense::update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
                   double lr) noexcept {
  W_ -= lr * dW;
  b_ -= lr * db;
}

} // namespace FlexNN::Layers
