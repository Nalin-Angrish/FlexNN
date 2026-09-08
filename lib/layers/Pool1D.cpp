/**
 * @file Pool1D.cpp
 * @brief Implements MaxPool1D and AvgPool1D forward/backward.
 */

#include "layers/Pool1D.hpp"

#include <cassert>
#include <limits>
#include <string>

#include "activations/detail.hpp"

namespace FlexNN::Layers {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int l_out_pool(int L_in, int K, int stride, int pad) noexcept {
  return (L_in + 2 * pad - K) / stride + 1;
}

// ---------------------------------------------------------------------------
// MaxPool1D
// ---------------------------------------------------------------------------

MaxPool1D::MaxPool1D(Pool1DParams p, Activations::Activation act)
    : params_(p), act_(act) {
  assert(p.channels > 0 && p.kernelSize > 0 && p.stride > 0);
}

MaxPool1D::MaxPool1D(Pool1DParams p, const std::string& actStr)
    : MaxPool1D(p, Activations::Activation::None) {
  Activations::Activation parsed;
  if (Activations::try_parse(actStr, parsed)) act_ = parsed;
}

int MaxPool1D::l_out(int L_in) const noexcept {
  return l_out_pool(L_in, params_.kernelSize, params_.stride, params_.padding);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> MaxPool1D::forward(
    const Eigen::MatrixXd& input) const {
  int C = params_.channels;
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  assert(input.rows() % C == 0);
  int L_in = static_cast<int>(input.rows() / C);
  int L_out = l_out(L_in);
  assert(L_out >= 1);
  int batch = static_cast<int>(input.cols());
  Eigen::MatrixXd Z(C * L_out, batch);
  Z.setZero();

  // Prepare cache for backward: [batch][C*L_out] -> arg index (in_pos)
  argmax_cache_.assign(batch, std::vector<int>(C * L_out, -1));
  last_L_in_ = L_in;
  last_batch_ = batch;

  for (int n = 0; n < batch; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int ol = 0; ol < L_out; ++ol) {
        double best = -std::numeric_limits<double>::infinity();
        int best_k = -1;
        for (int k = 0; k < K; ++k) {
          int in_pos = ol * s - p + k;
          double val = -std::numeric_limits<double>::infinity();
          if (in_pos >= 0 && in_pos < L_in) {
            int row = c * L_in + in_pos;
            val = input(row, n);
          }
          if (val > best) {
            best = val;
            best_k = k;
          }
        }
        int z_row = c * L_out + ol;
        Z(z_row, n) = best;
        // Store arg index for backward: which in_pos won
        int in_pos = ol * s - p + best_k;
        argmax_cache_[n][c * L_out + ol] = in_pos;
      }
    }
  }

  Eigen::MatrixXd A = Activations::detail::forward(act_, Z);
  return {Z, A};
}

Eigen::MatrixXd MaxPool1D::backward(const Eigen::MatrixXd& upstream,
                                    const Eigen::MatrixXd& currZ) const {
  Eigen::MatrixXd A = Activations::detail::forward(act_, currZ);
  return Activations::detail::backward(act_, upstream, currZ, A);
}

Eigen::MatrixXd MaxPool1D::backward(const Eigen::MatrixXd& nextW,
                                    const Eigen::MatrixXd& nextdZ,
                                    const Eigen::MatrixXd& currZ) const {
  Eigen::MatrixXd upstream;
  if (nextW.size() == 0) upstream = nextdZ;
  else upstream = nextW.transpose() * nextdZ;
  return backward(upstream, currZ);
}

Eigen::MatrixXd MaxPool1D::propagate(const Eigen::MatrixXd& dZ) const {
  int C = params_.channels;
  int L_out = static_cast<int>(dZ.rows() / C);
  int batch = static_cast<int>(dZ.cols());
  // Use cached L_in if available and batch matches, else derive
  int L_in = last_L_in_;
  if (last_batch_ != batch || L_in <= 0) {
    // Fallback: derive L_in from L_out inverse (approx)
    // L_in = (L_out-1)*stride + K -2*pad ; for Max we can compute
    L_in = (L_out - 1) * params_.stride + params_.kernelSize - 2 * params_.padding;
    if (L_in < 1) L_in = 1;
  }
  Eigen::MatrixXd dX(C * L_in, batch);
  dX.setZero();
  for (int n = 0; n < batch; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int ol = 0; ol < L_out; ++ol) {
        int z_row = c * L_out + ol;
        double dz = dZ(z_row, n);
        int in_pos = -1;
        if (n < static_cast<int>(argmax_cache_.size()) &&
            c * L_out + ol < static_cast<int>(argmax_cache_[n].size())) {
          in_pos = argmax_cache_[n][c * L_out + ol];
        }
        if (in_pos >= 0 && in_pos < L_in) {
          int x_row = c * L_in + in_pos;
          dX(x_row, n) += dz;
        }
      }
    }
  }
  return dX;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> MaxPool1D::grad(
    const Eigen::MatrixXd& /*dZ*/, const Eigen::MatrixXd& /*input*/) const {
  // Pool has no weights
  return {Eigen::MatrixXd(), Eigen::VectorXd()};
}

void MaxPool1D::update(const Eigen::MatrixXd& /*dW*/, const Eigen::VectorXd& /*db*/,
                       double /*lr*/) noexcept {}

// ---------------------------------------------------------------------------
// AvgPool1D
// ---------------------------------------------------------------------------

AvgPool1D::AvgPool1D(Pool1DParams p, Activations::Activation act)
    : params_(p), act_(act) {
  assert(p.channels > 0 && p.kernelSize > 0 && p.stride > 0);
}

AvgPool1D::AvgPool1D(Pool1DParams p, const std::string& actStr)
    : AvgPool1D(p, Activations::Activation::None) {
  Activations::Activation parsed;
  if (Activations::try_parse(actStr, parsed)) act_ = parsed;
}

int AvgPool1D::l_out(int L_in) const noexcept {
  return l_out_pool(L_in, params_.kernelSize, params_.stride, params_.padding);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AvgPool1D::forward(
    const Eigen::MatrixXd& input) const {
  int C = params_.channels;
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  assert(input.rows() % C == 0);
  int L_in = static_cast<int>(input.rows() / C);
  int L_out = l_out(L_in);
  int batch = static_cast<int>(input.cols());
  Eigen::MatrixXd Z(C * L_out, batch);
  Z.setZero();
  for (int n = 0; n < batch; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int ol = 0; ol < L_out; ++ol) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
          int in_pos = ol * s - p + k;
          if (in_pos >= 0 && in_pos < L_in) {
            int row = c * L_in + in_pos;
            sum += input(row, n);
          }
        }
        int z_row = c * L_out + ol;
        Z(z_row, n) = sum / static_cast<double>(K);
      }
    }
  }
  Eigen::MatrixXd A = Activations::detail::forward(act_, Z);
  return {Z, A};
}

Eigen::MatrixXd AvgPool1D::backward(const Eigen::MatrixXd& upstream,
                                    const Eigen::MatrixXd& currZ) const {
  Eigen::MatrixXd A = Activations::detail::forward(act_, currZ);
  return Activations::detail::backward(act_, upstream, currZ, A);
}

Eigen::MatrixXd AvgPool1D::backward(const Eigen::MatrixXd& nextW,
                                    const Eigen::MatrixXd& nextdZ,
                                    const Eigen::MatrixXd& currZ) const {
  Eigen::MatrixXd upstream;
  if (nextW.size() == 0) upstream = nextdZ;
  else upstream = nextW.transpose() * nextdZ;
  return backward(upstream, currZ);
}

Eigen::MatrixXd AvgPool1D::propagate(const Eigen::MatrixXd& dZ) const {
  int C = params_.channels;
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  int L_out = static_cast<int>(dZ.rows() / C);
  int batch = static_cast<int>(dZ.cols());
  // Derive L_in as inverse
  int L_in = (L_out - 1) * s + K - 2 * p;
  if (L_in < 1) L_in = 1;
  Eigen::MatrixXd dX(C * L_in, batch);
  dX.setZero();
  for (int n = 0; n < batch; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int ol = 0; ol < L_out; ++ol) {
        int z_row = c * L_out + ol;
        double dz = dZ(z_row, n) / static_cast<double>(K);
        for (int k = 0; k < K; ++k) {
          int in_pos = ol * s - p + k;
          if (in_pos >= 0 && in_pos < L_in) {
            int x_row = c * L_in + in_pos;
            dX(x_row, n) += dz;
          }
        }
      }
    }
  }
  return dX;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> AvgPool1D::grad(
    const Eigen::MatrixXd& /*dZ*/, const Eigen::MatrixXd& /*input*/) const {
  return {Eigen::MatrixXd(), Eigen::VectorXd()};
}

void AvgPool1D::update(const Eigen::MatrixXd& /*dW*/, const Eigen::VectorXd& /*db*/,
                       double /*lr*/) noexcept {}

} // namespace FlexNN::Layers
