/**
 * @file Pool1D.hpp
 * @brief Pool1D layers — MaxPool1D and AvgPool1D (weightless).
 *
 * Both share `Pool1DParams` (channels, kernelSize, stride, padding) and are
 * `Activation::None` only in v0.1. MaxPool routes gradient to argmax;
 * AvgPool distributes `1/K`. No weights, so `model.bin` has `w_cnt=b_cnt=0`.
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "LayerTypes.hpp"
#include "../activations/Activation.hpp"

namespace FlexNN::Layers {

class MaxPool1D {
 public:
  explicit MaxPool1D(Pool1DParams p,
                     Activations::Activation act = Activations::Activation::None);
  [[deprecated("use Activations::Activation enum")]]
  MaxPool1D(Pool1DParams p, const std::string& actStr);

  LayerType type() const noexcept { return LayerType::MaxPool1D; }
  Activations::Activation activation() const noexcept { return act_; }
  const Pool1DParams& params() const noexcept { return params_; }

  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input) const;
  Eigen::MatrixXd backward(const Eigen::MatrixXd& upstream,
                           const Eigen::MatrixXd& currZ) const;
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
  int l_out(int L_in) const noexcept;
  Pool1DParams params_{};
  Activations::Activation act_ = Activations::Activation::None;
  // Cache for backward: argmax indices per [C*L_out × batch] -> flat index
  mutable std::vector<std::vector<int>> argmax_cache_; // per batch, per output
  mutable int last_L_in_ = 0;
  mutable int last_batch_ = 0;
};

class AvgPool1D {
 public:
  explicit AvgPool1D(Pool1DParams p,
                     Activations::Activation act = Activations::Activation::None);
  [[deprecated("use Activations::Activation enum")]]
  AvgPool1D(Pool1DParams p, const std::string& actStr);

  LayerType type() const noexcept { return LayerType::AvgPool1D; }
  Activations::Activation activation() const noexcept { return act_; }
  const Pool1DParams& params() const noexcept { return params_; }

  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(
      const Eigen::MatrixXd& input) const;
  Eigen::MatrixXd backward(const Eigen::MatrixXd& upstream,
                           const Eigen::MatrixXd& currZ) const;
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
  int l_out(int L_in) const noexcept;
  Pool1DParams params_{};
  Activations::Activation act_ = Activations::Activation::None;
};

} // namespace FlexNN::Layers
