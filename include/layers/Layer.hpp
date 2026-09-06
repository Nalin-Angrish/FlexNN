/**
 * @file Layer.hpp
 * @brief Type-erased wrapper for FlexNN::Layers — heterogeneous stack without inheritance.
 *
 * Each concrete layer (Dense, Conv1D, BatchNorm1D, MaxPool1D, AvgPool1D) lives
 * in its own header (`include/layers/<Type>.hpp`) and owns its weights.
 * `FlexNN::Layers::Layer` is a thin wrapper over `std::variant` that lets
 * `FlexNN::NeuralNetwork` hold a `vector<Layer>` of mixed types while keeping
 * dispatch `std::visit` (no vtable, no heap, inline). New layer types are
 * added by expanding the variant in later PRs — existing `visit` sites
 * (forward/backward in `NeuralNetwork`, `ModelIO`) automatically handle them
 * via the same pattern without touching old concrete headers.
 *
 * Why variant, not inheritance (see LLD_FLEXNN §3):
 * - Host training stays simple (Eigen, no vtable indirection)
 * - For MCU we don't reuse these structs at all — export is flat arrays
 * - Variant is inline (no indirection) and `std::visit` is optimized away
 */

#pragma once

#include <type_traits>
#include <variant>

#include "Dense.hpp"
// Future layer headers will be included here as they are added:
// #include "Conv1D.hpp"
// #include "BatchNorm1D.hpp"
// #include "Pool1D.hpp"

namespace FlexNN::Layers {

/**
 * @brief Type-erased layer — holds one concrete layer type.
 *
 * In PR-04 the variant holds only `Dense`. Later PRs expand it:
 * - PR-05: `std::variant<Dense, Conv1D>`
 * - PR-07: `std::variant<Dense, Conv1D, BatchNorm1D>`
 * - PR-08: `std::variant<Dense, Conv1D, BatchNorm1D, MaxPool1D, AvgPool1D>`
 *
 * Each concrete type implements the same interface:
 *   LayerType type() const noexcept
 *   Activations::Activation activation() const noexcept
 *   pair<MatrixXd,MatrixXd> forward(const MatrixXd&) const
 *   MatrixXd backward(const MatrixXd& nextW, const MatrixXd& nextdZ, const MatrixXd& currZ) const
 *   void update(const MatrixXd& dW, const VectorXd& db, double lr) noexcept
 *
 * The wrapper forwards to the active alternative via `std::visit`.
 */
class Layer {
 public:
  // Construction from each concrete type — implicit so
  // `NeuralNetwork net({Dense(...), Dense(...)})` works.
  Layer(Dense d) : var_(std::move(d)) {}
  // Future ctors added in later PRs:
  // Layer(Conv1D c) : var_(std::move(c)) {}
  // Layer(BatchNorm1D b) : var_(std::move(b)) {}
  // Layer(MaxPool1D m) : var_(std::move(m)) {}
  // Layer(AvgPool1D a) : var_(std::move(a)) {}

  /**
   * @brief Layer kind (Dense, Conv1D, ...).
   */
  LayerType type() const noexcept {
    return std::visit([](auto&& v) { return v.type(); }, var_);
  }

  /**
   * @brief Activation for this layer.
   */
  Activations::Activation activation() const noexcept {
    return std::visit([](auto&& v) { return v.activation(); }, var_);
  }

  /**
   * @brief Forward pass for this layer (dispatch via visit).
   * @param input Batch input `[in_dim × batch]`
   * @return Pair (Z, A) each `[out_dim × batch]`
   */
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input) const {
    return std::visit([&](auto&& v) { return v.forward(input); }, var_);
  }

  /**
   * @brief Backward pass for this layer (dispatch via visit).
   *
   * For the last layer, `NeuralNetwork::backward` fuses Softmax+CE and never
   * calls this for the last layer. For hidden layers this applies
   * `f'(Z) ⊙ (W_next^T * dZ_next)`.
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& nextW,
                           const Eigen::MatrixXd& nextdZ,
                           const Eigen::MatrixXd& currZ) const {
    return std::visit(
        [&](auto&& v) { return v.backward(nextW, nextdZ, currZ); }, var_);
  }

  /**
   * @brief SGD update — dispatches to concrete `update`.
   *
   * Kept as `updateWeights` for compatibility with legacy `Layer::updateWeights`
   * signature used by `NeuralNetwork::updateWeights`. New code may call `update`.
   */
  void updateWeights(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
                     double lr) noexcept {
    std::visit([&](auto&& v) { v.update(dW, db, lr); }, var_);
  }
  void update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
              double lr) noexcept {
    updateWeights(dW, db, lr);
  }

  /**
   * @brief Weight matrix for backward (W_next^T * dZ_next).
   *
   * For Dense this is `W [out×in]`; for weightless layers (Pool/BN) this
   * would be empty in future PRs and the caller handles pooling/BN
   * specially. In PR-04 only Dense exists, so this always returns Dense weights.
   */
  Eigen::MatrixXd weightsMatrix() const noexcept {
    return std::visit(
        [](auto&& v) -> Eigen::MatrixXd {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Dense>) {
            return v.weights();
          } else {
            return Eigen::MatrixXd(); // future weightless layers
          }
        },
        var_);
  }

  /**
   * @brief Direct access to the underlying variant (for ModelIO).
   *
   * Lets `ModelIO` do `std::visit` without public variant exposure of
   * concrete types. Kept const — mutation goes via `updateWeights`.
   */
  const auto& variant() const noexcept { return var_; }
  auto& variant() noexcept { return var_; }

  // Helpers for ModelIO/tests to query concrete type without visit boilerplate.
  // Returns nullptr if the active alternative is not Dense.
  const Dense* asDense() const noexcept {
    return std::get_if<Dense>(&var_);
  }
  Dense* asDense() noexcept { return std::get_if<Dense>(&var_); }

 private:
  // In PR-04 only Dense is active. This keeps the variant size minimal and
  // the stack base builds without pulling Conv1D/BatchNorm/Pool headers.
  std::variant<Dense> var_;
};

} // namespace FlexNN::Layers
