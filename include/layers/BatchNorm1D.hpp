/**
 * @file BatchNorm1D.hpp
 * @brief BatchNorm1D layer for FlexNN::Layers (training stats, aux export).
 *
 * Implements per-feature `y = gamma*(x - mean)/sqrt(var+eps) + beta` where
 * training uses batch stats and `predict` uses running stats (momentum 0.1).
 * The layer is `Activation::None` only in v0.1 (export error otherwise).
 * `aux` blob in `model.bin` is `[gamma, beta, mean, var]` each `[F]` float32.
 *
 * Why this file: BatchNorm is needed for stable Conv1D training and is
 * folded by the runtime compiler (`W_fold = W*gamma/sqrt(var+eps)`), but
 * FlexNN still exports it as a separate layer for training correctness.
 */

#pragma once

#include <string>
#include <utility>

#include <Eigen/Dense>

#include "LayerTypes.hpp"
#include "../activations/Activation.hpp"

namespace FlexNN::Layers {

class BatchNorm1D {
 public:
  explicit BatchNorm1D(BatchNormParams p,
                       Activations::Activation act = Activations::Activation::None);
  explicit BatchNorm1D(int numFeatures, float epsilon = 1e-5f,
                       Activations::Activation act = Activations::Activation::None);

  [[deprecated("use Activations::Activation enum")]]
  BatchNorm1D(BatchNormParams p, const std::string& actStr);
  [[deprecated("use Activations::Activation enum")]]
  BatchNorm1D(int numFeatures, const std::string& actStr);

  LayerType type() const noexcept { return LayerType::BatchNorm1D; }
  Activations::Activation activation() const noexcept { return act_; }
  const BatchNormParams& params() const noexcept { return params_; }

  // Accessors for ModelIO and tests (gamma/beta/mean/var each [F])
  const Eigen::VectorXd& gamma() const noexcept { return gamma_; }
  const Eigen::VectorXd& beta() const noexcept { return beta_; }
  const Eigen::VectorXd& runningMean() const noexcept { return runningMean_; }
  const Eigen::VectorXd& runningVar() const noexcept { return runningVar_; }
  void setGamma(const Eigen::VectorXd& g) { gamma_ = g; }
  void setBeta(const Eigen::VectorXd& b) { beta_ = b; }
  void setRunningMean(const Eigen::VectorXd& m) { runningMean_ = m; }
  void setRunningVar(const Eigen::VectorXd& v) { runningVar_ = v; }

  /**
   * @brief Forward with batch stats (training) — updates running stats.
   *
   * Uses batch mean/var per feature (row) across batch*L_out? For 1D case
   * with flattened [F*L × batch] input, we treat each row as a feature
   * and compute mean/var over cols (batch). This matches the flattened
   * Dense/BatchNorm stack where each feature is a separate row.
   *
   * @param input [F × batch] or [F*L × batch] (each row is a feature)
   * @return Pair (Z, A) where Z is normalized `y` and A = act(Z) (None)
   */
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input) const;

  /**
   * @brief Forward with explicit training flag.
   *
   * When `training==false`, uses runningMean/Var (for predict).
   */
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input, bool training) const;

  // Backward with upstream (for hidden layers)
  Eigen::MatrixXd backward(const Eigen::MatrixXd& upstream,
                           const Eigen::MatrixXd& currZ) const;
  // Legacy 3-arg for compatibility (delegates to upstream version)
  Eigen::MatrixXd backward(const Eigen::MatrixXd& nextW,
                           const Eigen::MatrixXd& nextdZ,
                           const Eigen::MatrixXd& currZ) const;

  Eigen::MatrixXd propagate(const Eigen::MatrixXd& dZ) const;

  std::pair<Eigen::MatrixXd, Eigen::VectorXd> grad(
      const Eigen::MatrixXd& dZ, const Eigen::MatrixXd& input) const;

  void update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
              double lr) noexcept;
  void updateWeights(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
                     double lr) noexcept {
    update(dW, db, lr);
  }

 private:
  BatchNormParams params_{};
  Activations::Activation act_ = Activations::Activation::None;
  Eigen::VectorXd gamma_;       ///< [F]
  Eigen::VectorXd beta_;        ///< [F]
  mutable Eigen::VectorXd runningMean_; ///< [F] mutable so forward can update
  mutable Eigen::VectorXd runningVar_;  ///< [F]
};

} // namespace FlexNN::Layers
