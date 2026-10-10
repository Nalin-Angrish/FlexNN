/**
 * @file FlexNN.h
 * @brief Header for FlexNN neural network (heterogeneous stack via Layers::Layer).
 *
 * After PR-04, `NeuralNetwork` holds `std::vector<FlexNN::Layers::Layer>` —
 * a type-erased variant over `Dense` (and later `Conv1D`, `BatchNorm1D`,
 * `MaxPool1D`, `AvgPool1D`). The legacy `FlexNN::Layer` (stringly-typed Dense)
 * is kept for one release as a deprecated shim so `src/main.cpp` still
 * builds; it is converted to `Layers::Dense` via `Activations::try_parse`.
 *
 * Why variant, not inheritance (see LLD_FLEXNN §3): host training stays
 * simple (Eigen, no vtable), variant is inline, and `std::visit` dispatches
 * without indirection. For MCU we don't reuse these structs at all — export
 * is flat arrays.
 */

#ifndef FlexNN_H
#define FlexNN_H

#include <vector>
#include <Eigen/Dense>

// New modular headers — must come before legacy Layer.h so `Layers::Layer`
// is visible for the converting constructor.
#include "activations/Activation.hpp"
#include "layers/Layer.hpp"

// Legacy monolithic Layer (string activation, Dense-only) — kept for
// backward compat until PR-09 migrates `src/main.cpp` to the new API.
// New code should use `FlexNN::Layers::Dense` + `FlexNN::Activations::Activation`.
#include "Layer.h"

namespace FlexNN {

/**
 * @class NeuralNetwork
 * @brief Linear stack of heterogeneous layers (vector<Layers::Layer>).
 *
 * Currently holds only `Layers::Dense` (PR-04). Later PRs expand the variant
 * to `Conv1D`, `BatchNorm1D`, `MaxPool1D`, `AvgPool1D` without changing this
 * header — only `include/layers/Layer.hpp`'s variant list grows.
 *
 * Old code `NeuralNetwork({Layer(784,64,"relu"), Layer(64,10,"softmax")})`
 * still compiles via the deprecated converting constructor that maps each
 * legacy `Layer` to `Layers::Dense` via `Activations::try_parse`.
 */
class NeuralNetwork {
 public:
  /**
   * @brief Construct from new modular layers (preferred).
   *
   * @param layers Heterogeneous stack — each element is a `Layers::Layer`
   *               wrapping a concrete `Dense`/`Conv1D`/...
   *               Example: `NeuralNetwork({Layers::Dense(784,64, Activations::ReLU)})`
   */
  NeuralNetwork(const std::vector<Layers::Layer>& layers) : layers_(layers) {}
  NeuralNetwork(std::vector<Layers::Layer>&& layers) : layers_(std::move(layers)) {}

  /**
   * @brief Deprecated: construct from legacy `FlexNN::Layer` (Dense-only).
   *
   * Converts each legacy `Layer` (which stores `std::string activationFunction`
   * and `MatrixXd W`/`VectorXd b`) to `Layers::Dense` via `Activations::try_parse`.
   * On unknown activation string, uses `Activation::None`.
   *
   * Kept so `src/main.cpp` (which still uses `Layer(..., "relu")`) builds
   * until PR-09 migrates it to `Layers::Dense`. New code should not use this.
   */
  [[deprecated("use vector<FlexNN::Layers::Layer> with FlexNN::Activations")]]
  NeuralNetwork(const std::vector<Layer>& oldLayers);

  /**
   * @brief Train the network on (X, Y) with SGD.
   *
   * `X` is `[features × batch]` (see Utility.h), `Y` is label vector
   * (converted to one-hot inside). Uses the interleaved `forward` layout
   * `[input, Z0,A0, Z1,A1, ...]` and `backward` that fuses `Softmax+CE` as
   * `dZ_last = (A - Y)/m`.
   */
  void train(const Eigen::MatrixXd& input, const Eigen::VectorXd& target,
             double learningRate, int epochs);

  /**
   * @brief Accuracy on (X, Y) — argmax per column.
   */
  double accuracy(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y);

  /**
   * @brief Predict — forward pass, returns activation of last layer.
   */
  Eigen::MatrixXd predict(const Eigen::MatrixXd& input) {
    auto outputs = forward(input);
    return outputs.back();
  }

  /**
   * @brief Direct access to layers (for tests / ModelIO).
   */
  const std::vector<Layers::Layer>& layers() const noexcept { return layers_; }
  std::vector<Layers::Layer>& layers() noexcept { return layers_; }

 private:
  // Heterogeneous stack — currently variant<Dense> only, expands in later PRs.
  std::vector<Layers::Layer> layers_;

  /**
   * @brief Forward pass through all layers.
   * @return Vector `[input, Z0,A0, Z1,A1, ...]` (size `1 + 2*layers_.size()`)
   */
  std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input);

  /**
   * @brief Backward pass — computes dZ per layer.
   * @return Gradients as `[dW0, db0, dW1, db1, ...]` (size `2*layers_.size()`)
   *         in layer order (reversed inside then flipped).
   */
  std::vector<Eigen::MatrixXd> backward(
      const std::vector<Eigen::MatrixXd>& outputs,
      const Eigen::MatrixXd& target);

  /**
   * @brief SGD update from gradients.
   */
  void updateWeights(const std::vector<Eigen::MatrixXd>& gradients,
                     double learningRate);
};

} // namespace FlexNN

#endif // FlexNN_H
