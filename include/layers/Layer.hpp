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
#include "Conv1D.hpp"
#include "BatchNorm1D.hpp"
#include "activations/ActivationParameters.hpp"
// Future layer headers will be included here as they are added:
// #include "Pool1D.hpp"

namespace FlexNN::Layers {

/**
 * @brief Type-erased layer — holds one concrete layer type.
 *
 * In PR-05 the variant holds `Dense` and `Conv1D`. Later PRs expand it:
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
  Layer(Conv1D c) : var_(std::move(c)) {}
  Layer(BatchNorm1D b) : var_(std::move(b)) {}
  // Future ctors added in later PRs:
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
   * @brief Activation hyperparameters (e.g., LeakyReLU alpha).
   *
   * Returns stored `ActivationParameters` for layers that carry it
   * (`Dense`, `Conv1D`); otherwise default. Non-Leaky activations ignore the
   * returned struct.
   */
  Activations::ActivationParameters activationParams() const noexcept {
    return std::visit(
        [](auto&& v) -> Activations::ActivationParameters {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Dense> || std::is_same_v<T, Conv1D>) {
            return v.activationParams();
          } else {
            return Activations::ActivationParameters{};
          }
        },
        var_);
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
   *
   * Kept for `dense_test` compatibility (takes nextW/nextdZ).
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& nextW,
                           const Eigen::MatrixXd& nextdZ,
                           const Eigen::MatrixXd& currZ) const {
    return std::visit(
        [&](auto&& v) { return v.backward(nextW, nextdZ, currZ); }, var_);
  }

  /**
   * @brief Backward with already-propagated upstream (W_next^T*dZ_next).
   *
   * Used by `NeuralNetwork::backward` after `layers[i+1].propagate(dZ_next)`
   * so the caller handles `Conv1D` transpose via `col2im`.
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& upstream,
                           const Eigen::MatrixXd& currZ) const {
    return std::visit(
        [&](auto&& v) { return v.backward(upstream, currZ); }, var_);
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
   * @brief Weight matrix for backward — Dense/Conv1D.
   *
   * For Dense this is `W [out×in]`; for Conv1D this is `W [outCh×inCh*K]`.
   * Kept for ModelIO inspection; for backward prefer `propagate()` which
   * handles Conv1D col2im correctly. Not noexcept (allocates on copy).
   */
  Eigen::MatrixXd weightsMatrix() const {
    return std::visit(
        [](auto&& v) -> Eigen::MatrixXd {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Dense>) {
            return v.weights();
          } else if constexpr (std::is_same_v<T, Conv1D>) {
            return v.weights();
          } else if constexpr (std::is_same_v<T, BatchNorm1D>) {
            return Eigen::MatrixXd(); // BN is weightless for W^T* dZ (handled via propagate)
          } else {
            return Eigen::MatrixXd();
          }
        },
        var_);
  }

  /**
   * @brief Propagate gradient to previous layer (W^T * dZ or col2im).
   *
   * Used by `NeuralNetwork::backward` to compute upstream for hidden layers:
   * `upstream = layers[i+1].propagate(dZ_next)`. For Dense this is
   * `W^T * dZ`; for Conv1D this is `col2im(W^T * dZ)` via `Conv1D::propagate`.
   *
   * @note Not noexcept: allocates MatrixXd (host, bad_alloc terminates).
   */
  Eigen::MatrixXd propagate(const Eigen::MatrixXd& dZ) const {
    return std::visit(
        [&](auto&& v) -> Eigen::MatrixXd {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Dense>) {
            return v.weights().transpose() * dZ;
          } else if constexpr (std::is_same_v<T, Conv1D>) {
            return v.propagate(dZ);
          } else if constexpr (std::is_same_v<T, BatchNorm1D>) {
            return v.propagate(dZ);
          } else {
            return dZ;
          }
        },
        var_);
  }

  /**
   * @brief Compute dW/db for this layer given dZ and input.
   *
   * For Dense: `dW = dZ * X^T / batch`, `db = rowMean(dZ)`.
   * For Conv1D: `dW/db` via `Conv1D::grad`.
   * Returns pair (dW, db) as two matrices (db as column vector promoted to MatrixXd).
   */
  std::pair<Eigen::MatrixXd, Eigen::VectorXd> grad(
      const Eigen::MatrixXd& dZ, const Eigen::MatrixXd& input) const {
    return std::visit(
        [&](auto&& v) -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Dense>) {
            double m = static_cast<double>(dZ.cols());
            Eigen::MatrixXd dW = dZ * input.transpose() / m;
            Eigen::VectorXd db = dZ.rowwise().mean();
            return {dW, db};
          } else if constexpr (std::is_same_v<T, Conv1D>) {
            return v.grad(dZ, input);
          } else if constexpr (std::is_same_v<T, BatchNorm1D>) {
            return v.grad(dZ, input);
          } else {
            return {Eigen::MatrixXd(), Eigen::VectorXd()};
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
  const Dense* asDense() const noexcept { return std::get_if<Dense>(&var_); }
  Dense* asDense() noexcept { return std::get_if<Dense>(&var_); }
  const Conv1D* asConv1D() const noexcept { return std::get_if<Conv1D>(&var_); }
  Conv1D* asConv1D() noexcept { return std::get_if<Conv1D>(&var_); }
  const BatchNorm1D* asBatchNorm1D() const noexcept {
    return std::get_if<BatchNorm1D>(&var_);
  }
  BatchNorm1D* asBatchNorm1D() noexcept {
    return std::get_if<BatchNorm1D>(&var_);
  }

 private:
  std::variant<Dense, Conv1D, BatchNorm1D> var_;
};

} // namespace FlexNN::Layers
