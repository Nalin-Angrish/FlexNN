/**
 * @file Conv1D.hpp
 * @brief 1D temporal convolution for FlexNN::Layers.
 *
 * Implements `Conv1D` as described in LLD_FLEXNN §6.2. The layer operates
 * on flattened `[C*L × batch]` matrices where `C` is channels and `L` is
 * temporal length. Forward is zero-padded, backward uses im2col/col2im
 * for `dW` and `dX`. Dilation !=1 is rejected at export (Status error)
 * but training still works (forward handles dilation).
 *
 * Weight layout: `W [outChannels × inChannels*K]` row-major logical,
 * stored in Eigen column-major `MatrixXd` but serialized row-major.
 * Bias `b [outChannels]` broadcast over `L_out`.
 */

#pragma once

#include <string>
#include <utility>

#include <Eigen/Dense>

#include "LayerTypes.hpp"
#include "activations/Activation.hpp"

namespace FlexNN::Layers {

class Conv1D {
 public:
  /**
   * @brief Construct from params and activation.
   *
   * Initializes `W` and `b` with `Random * 0.5` (same as Dense for v0.1).
   * @param p Conv1DParams with inChannels/outChannels/kernelSize/stride/pad/dilation
   * @param act Activation (default ReLU)
   */
  explicit Conv1D(Conv1DParams p,
                  Activations::Activation act = Activations::Activation::ReLU);

  /**
   * @brief Convenience ctor with explicit ints.
   */
  Conv1D(int inChannels, int outChannels, int kernelSize, int stride = 1,
         int padding = 0, int dilation = 1,
         Activations::Activation act = Activations::Activation::ReLU);

  [[deprecated("use Activations::Activation enum")]]
  Conv1D(Conv1DParams p, const std::string& actStr);
  [[deprecated("use Activations::Activation enum")]]
  Conv1D(int inChannels, int outChannels, int kernelSize, int stride,
         int padding, int dilation, const std::string& actStr);

  LayerType type() const noexcept { return LayerType::Conv1D; }
  Activations::Activation activation() const noexcept { return act_; }
  const Conv1DParams& params() const noexcept { return params_; }
  const Eigen::MatrixXd& weights() const noexcept { return W_; }
  const Eigen::VectorXd& biases() const noexcept { return b_; }

  void setWeights(const Eigen::MatrixXd& W) { W_ = W; }
  void setBiases(const Eigen::VectorXd& b) { b_ = b; }

  /**
   * @brief Forward: `Z[oc][ol] = b[oc] + Σic Σk W[oc][ic*K+k] * Xpad[ic][ol*stride - pad + k*dilation]`
   *
   * @param input Flattened input `[C_in*L_in × batch]` (each column is a sample)
   * @return Pair (Z, A) each `[C_out*L_out × batch]`
   */
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input) const;

  /**
   * @brief Backward for hidden layers (old signature, for dense_test compatibility).
   *
   * Computes `dA = W_next^T * dZ_next` then `dZ = f'(Z) ⊙ dA`.
   * Kept for `Dense`-style testing; new code should use `backward(upstream, currZ)`.
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& nextW,
                           const Eigen::MatrixXd& nextdZ,
                           const Eigen::MatrixXd& currZ) const;

  /**
   * @brief Backward with already-propagated upstream gradient.
   *
   * @param upstream Gradient w.r.t. A of this layer (already `W_next^T*dZ_next` or `propagate` result)
   * @param currZ Pre-activation Z of this layer from forward
   * @return Gradient w.r.t. Z of this layer
   */
  Eigen::MatrixXd backward(const Eigen::MatrixXd& upstream,
                           const Eigen::MatrixXd& currZ) const;

  /**
   * @brief Propagate gradient to previous layer: `dX = W^T * dZ` via col2im.
   *
   * For Conv1D, this is the transpose convolution (col2im). Input `dZ` is
   * `[C_out*L_out × batch]`, output `dX` is `[C_in*L_in × batch]` where
   * `L_in` is derived from `dZ`'s implied `L_out` and params.
   * The caller must know `L_in` — we derive it from `dZ` rows and params
   * assuming the forward `L_out` formula. For the common case where Conv1D
   * is followed by Dense, `L_in` is not needed because Dense's `dZ` is
   * already flattened; this method is used when Conv1D is the *next* layer
   * for a previous Dense — the Dense's upstream is `propagate(dZ_next)` from Conv1D.
   *
   * @param dZ Gradient w.r.t. Z of this Conv1D layer `[C_out*L_out × batch]`
   * @return Gradient w.r.t. input of this layer `[C_in*L_in × batch]` (derived)
   */
  Eigen::MatrixXd propagate(const Eigen::MatrixXd& dZ) const;

  /**
   * @brief Compute gradients dW and db for SGD update.
   *
   * Uses im2col: `col [C_in*K × L_out]` per sample, `dW = Σ_n dZ_n * col_n^T / batch`,
   * `db = rowMean over (batch*L_out)` per channel (mean over batch and spatial).
   *
   * @param dZ Gradient w.r.t. Z `[C_out*L_out × batch]`
   * @param input Original input to forward `[C_in*L_in × batch]`
   * @return Pair (dW, db) where dW is `[outChannels × inChannels*K]`, db is `[outChannels]`
   */
  std::pair<Eigen::MatrixXd, Eigen::VectorXd> grad(
      const Eigen::MatrixXd& dZ, const Eigen::MatrixXd& input) const;

  void update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
              double lr) noexcept;
  void updateWeights(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
                     double lr) noexcept {
    update(dW, db, lr);
  }

 private:
  // Helper: compute L_out from L_in
  int l_out(int L_in) const noexcept;
  // Helper: compute L_in from L_out (inverse, for propagate)
  int l_in_from_lout(int L_out) const noexcept;

  Conv1DParams params_{};
  Activations::Activation act_ = Activations::Activation::ReLU;
  Eigen::MatrixXd W_; ///< [outChannels × inChannels*K]
  Eigen::VectorXd b_; ///< [outChannels]
};

} // namespace FlexNN::Layers
