/**
 * @file Dense.hpp
 * @brief Dense (fully-connected) layer for FlexNN::Layers.
 *
 * Each `Dense` instance owns its weights `W [out×in]` and biases `b [out]`
 * as `double` (Eigen `MatrixXd`/`VectorXd`) for training. Export via
 * `ModelIO` casts to `float32` row-major. The layer is stateless aside
 * from its parameters and activation — no virtual, no heap beyond Eigen.
 */

#pragma once

#include <string>
#include <utility>

#include <Eigen/Dense>

#include "LayerTypes.hpp"
#include "activations/Activation.hpp"
#include "activations/ActivationParameters.hpp"

namespace FlexNN::Layers {

/**
 * @brief Fully-connected layer: `Z = W*X + b`, `A = act(Z)`.
 *
 * Layout: `X` is `[in × batch]` column-major, `W` is `[out × in]`,
 * `b` is `[out]` broadcast column-wise. Activation is applied
 * elementwise via `FlexNN::Activations::detail`.
 */
class Dense {
 public:
  /**
   * @brief Construct Dense with input/output sizes and activation.
   *
   * Initializes `W` and `b` with `Random * 0.5` (same as legacy
   * `include/Layer.h:55`). Kept for v0.1 compatibility; future
   * versions may use Xavier/He initialization.
   *
   * @param in Number of input features
   * @param out Number of output neurons
   * @param act Activation (default ReLU)
   */
  Dense(int in, int out,
        Activations::Activation act = Activations::Activation::ReLU,
        const Activations::ActivationParameters& params = Activations::ActivationParameters{});

  /**
   * @brief Construct from DenseParams POD.
   */
  explicit Dense(DenseParams p,
                 Activations::Activation act = Activations::Activation::ReLU,
                 const Activations::ActivationParameters& params = Activations::ActivationParameters{});

  /**
   * @brief Deprecated string ctor — delegates to enum via try_parse.
   *
   * Kept for one release so `Dense(784,64,"relu")` still compiles with a
   * deprecation warning. On parse failure, activation is `None`.
   */
  [[deprecated("use Activations::Activation enum")]]
  Dense(int in, int out, const std::string& actStr);

  // Accessors — used by NeuralNetwork and ModelIO via visit.
  LayerType type() const noexcept { return LayerType::Dense; }
  Activations::Activation activation() const noexcept { return act_; }
  const Activations::ActivationParameters& activationParams() const noexcept { return actParams_; }
  const DenseParams& params() const noexcept { return params_; }
  const Eigen::MatrixXd& weights() const noexcept { return W_; }
  const Eigen::VectorXd& biases() const noexcept { return b_; }

  // Mutators for importModel (replaces weights after allocation).
  void setWeights(const Eigen::MatrixXd& W) { W_ = W; }
  void setBiases(const Eigen::VectorXd& b) { b_ = b; }
  void setActivationParameters(const Activations::ActivationParameters& p) { actParams_ = p; }

  /**
   * @brief Forward pass: `Z = W*X + b`, `A = act(Z)`.
   *
   * @param input Batch input `[in × batch]`
   * @return Pair (Z, A) each `[out × batch]`
   */
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input) const;

  /**
   * @brief Backward: `dZ = f'(Z) ⊙ (W_next^T * dZ_next)` or `dZ = W_next^T*dZ_next` if None.
   *
   * For Softmax hidden, this uses `detail::softmax_backward_hidden` via
   * `Activations::detail::backward`. Last-layer Softmax is fused in
   * `NeuralNetwork::backward` and never calls this for the last layer.
   * `activationParams` threaded for LeakyReLU alpha.
   *
   * @param nextW Weights of next layer (or empty for last layer)
   * @param nextdZ Gradient of next layer's Z
   * @param currZ This layer's pre-activation Z from forward
   * @return Gradient w.r.t. this layer's Z
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& nextW,
                           const Eigen::MatrixXd& nextdZ,
                           const Eigen::MatrixXd& currZ) const;

  /**
   * @brief Backward with already-propagated upstream (W_next^T*dZ_next).
   *
   * Used by `NeuralNetwork::backward` after `layers[i+1].propagate(dZ_next)`
   * so the caller handles `Conv1D` transpose via `col2im`. Equivalent to the
   * three-arg version but skips the `nextW` multiply.
   *
   * @param upstream Already `W_next^T * dZ_next` (or `propagate` result)
   * @param currZ This layer's Z
   * @return Gradient w.r.t. Z
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& upstream,
                           const Eigen::MatrixXd& currZ) const;

  /**
   * @brief Propagate gradient to previous layer: `dX = W^T * dZ`.
   *
   * @param dZ Gradient w.r.t. Z of this layer `[out × batch]`
   * @return Gradient w.r.t. input `[in × batch]`
   */
  Eigen::MatrixXd propagate(const Eigen::MatrixXd& dZ) const;

  /**
   * @brief SGD update: `W -= lr*dW`, `b -= lr*db`.
   */
  void update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
              double lr) noexcept;

 private:
  DenseParams params_{};
  Activations::Activation act_ = Activations::Activation::ReLU;
  Activations::ActivationParameters actParams_{};
  Eigen::MatrixXd W_; ///< [out × in] double
  Eigen::VectorXd b_; ///< [out] double
};

} // namespace FlexNN::Layers
