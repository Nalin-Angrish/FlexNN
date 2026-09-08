/**
 * @file Conv1D.cpp
 * @brief Implements FlexNN::Layers::Conv1D forward/backward/propagate.
 *
 * Forward and backward are naive loops (not im2col-GEMM) for clarity and
 * correctness; they are `double` and `OpenMP`-parallelizable in future PRs.
 * The logic matches LLD_FLEXNN §6.2 and is verified by `tests/conv1d_test.cpp`
 * via brute force and finite differences.
 */

#include "layers/Conv1D.hpp"

#include <cassert>
#include <cmath>
#include <string>

#include "activations/detail.hpp"

namespace FlexNN::Layers {

Conv1D::Conv1D(Conv1DParams p, Activations::Activation act,
             const Activations::ActivationParameters& params)
    : params_(p), act_(act), actParams_(params) {
  int K = params_.kernelSize;
  int inCh = params_.inChannels;
  int outCh = params_.outChannels;
  assert(K > 0 && inCh > 0 && outCh > 0 && "Conv1D dims must be positive");
  W_ = Eigen::MatrixXd::Random(outCh, inCh * K) * 0.5;
  b_ = Eigen::VectorXd::Random(outCh) * 0.5;
}

Conv1D::Conv1D(int inCh, int outCh, int K, int stride, int pad, int dilation,
               Activations::Activation act,
               const Activations::ActivationParameters& params)
    : Conv1D(Conv1DParams{inCh, outCh, K, stride, pad, dilation}, act, params) {}

Conv1D::Conv1D(Conv1DParams p, const std::string& actStr)
    : Conv1D(p, Activations::Activation::ReLU) {
  Activations::Activation parsed;
  if (Activations::try_parse(actStr, parsed)) act_ = parsed;
  else act_ = Activations::Activation::None;
}

Conv1D::Conv1D(int inCh, int outCh, int K, int stride, int pad, int dilation,
               const std::string& actStr)
    : Conv1D(Conv1DParams{inCh, outCh, K, stride, pad, dilation},
             Activations::Activation::ReLU) {
  Activations::Activation parsed;
  if (Activations::try_parse(actStr, parsed)) act_ = parsed;
  else act_ = Activations::Activation::None;
}

int Conv1D::l_out(int L_in) const noexcept {
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  int d = params_.dilation;
  // (L_in + 2*pad - dilation*(K-1) -1)/stride +1
  return (L_in + 2 * p - d * (K - 1) - 1) / s + 1;
}

int Conv1D::l_in_from_lout(int L_out) const noexcept {
  // Inverse of above: L_in = (L_out-1)*stride + dilation*(K-1) +1 -2*pad
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  int d = params_.dilation;
  return (L_out - 1) * s + d * (K - 1) + 1 - 2 * p;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Conv1D::forward(
    const Eigen::MatrixXd& input) const {
  // Input: [C_in*L_in × batch] flattened, column-major batch
  int inCh = params_.inChannels;
  int outCh = params_.outChannels;
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  int d = params_.dilation;

  assert(input.rows() % inCh == 0 && "input rows must be multiple of inChannels");
  int L_in = static_cast<int>(input.rows() / inCh);
  int L_out = l_out(L_in);
  assert(L_out >= 1 && "L_out must be >=1 check padding/stride");
  int batch = static_cast<int>(input.cols());

  Eigen::MatrixXd Z(outCh * L_out, batch);
  Z.setZero();

  // For each sample, each output channel, each output position
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < outCh; ++oc) {
      for (int ol = 0; ol < L_out; ++ol) {
        double sum = b_(oc);
        for (int ic = 0; ic < inCh; ++ic) {
          for (int k = 0; k < K; ++k) {
            int in_pos = ol * s - p + k * d;
            double xval = 0.0;
            if (in_pos >= 0 && in_pos < L_in) {
              // input is [C_in*L_in × batch], row = ic*L_in + in_pos
              int row = ic * L_in + in_pos;
              xval = input(row, n);
            }
            // W is [outCh × inCh*K], col = ic*K + k
            int col = ic * K + k;
            sum += W_(oc, col) * xval;
          }
        }
        int z_row = oc * L_out + ol;
        Z(z_row, n) = sum;
      }
    }
  }

  Eigen::MatrixXd A = Activations::detail::forward(act_, Z, actParams_);
  return {Z, A};
}

Eigen::MatrixXd Conv1D::backward(const Eigen::MatrixXd& upstream,
                                 const Eigen::MatrixXd& currZ) const {
  // Hidden backward: dZ = f'(Z) ⊙ upstream with per-layer params
  Eigen::MatrixXd A = Activations::detail::forward(act_, currZ, actParams_);
  return Activations::detail::backward(act_, upstream, currZ, A, actParams_);
}

Eigen::MatrixXd Conv1D::backward(const Eigen::MatrixXd& nextW,
                                 const Eigen::MatrixXd& nextdZ,
                                 const Eigen::MatrixXd& currZ) const {
  // Legacy signature kept for Dense-style testing: dA = W_next^T * dZ_next
  Eigen::MatrixXd upstream;
  if (nextW.size() == 0) {
    upstream = nextdZ;
  } else {
    upstream = nextW.transpose() * nextdZ;
  }
  return backward(upstream, currZ);
}

Eigen::MatrixXd Conv1D::propagate(const Eigen::MatrixXd& dZ) const {
  // Propagate gradient to previous layer's input: dX = W^T * dZ via col2im
  int inCh = params_.inChannels;
  int outCh = params_.outChannels;
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  int d = params_.dilation;

  assert(dZ.rows() % outCh == 0 && "dZ rows must be multiple of outChannels");
  int L_out = static_cast<int>(dZ.rows() / outCh);
  int batch = static_cast<int>(dZ.cols());
  int L_in = l_in_from_lout(L_out);
  assert(L_in >= 1);

  Eigen::MatrixXd dX(inCh * L_in, batch);
  dX.setZero();

  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < outCh; ++oc) {
      for (int ol = 0; ol < L_out; ++ol) {
        int z_row = oc * L_out + ol;
        double dz = dZ(z_row, n);
        for (int ic = 0; ic < inCh; ++ic) {
          for (int k = 0; k < K; ++k) {
            int in_pos = ol * s - p + k * d;
            if (in_pos >= 0 && in_pos < L_in) {
              int x_row = ic * L_in + in_pos;
              int w_col = ic * K + k;
              dX(x_row, n) += W_(oc, w_col) * dz;
            }
          }
        }
      }
    }
  }
  return dX;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Conv1D::grad(
    const Eigen::MatrixXd& dZ, const Eigen::MatrixXd& input) const {
  int inCh = params_.inChannels;
  int outCh = params_.outChannels;
  int K = params_.kernelSize;
  int s = params_.stride;
  int p = params_.padding;
  int d = params_.dilation;

  assert(input.rows() % inCh == 0);
  int L_in = static_cast<int>(input.rows() / inCh);
  int L_out = l_out(L_in);
  assert(dZ.rows() == outCh * L_out);
  assert(dZ.cols() == input.cols());
  int batch = static_cast<int>(input.cols());

  Eigen::MatrixXd dW(outCh, inCh * K);
  dW.setZero();
  Eigen::VectorXd db(outCh);
  db.setZero();

  // db = mean over (batch * L_out) per channel: rowMean over flattened dZ's channel groups
  // dZ is [C_out*L_out × batch]; for each oc, collect its L_out rows across batch
  for (int oc = 0; oc < outCh; ++oc) {
    double sum = 0.0;
    for (int n = 0; n < batch; ++n) {
      for (int ol = 0; ol < L_out; ++ol) {
        int row = oc * L_out + ol;
        sum += dZ(row, n);
      }
    }
    db(oc) = sum / (static_cast<double>(batch * L_out));
  }

  // dW via im2col: dW[oc][ic*K+k] = Σ_n Σ_ol dZ[oc][ol]_n * Xpad[ic][ol*s - p + k*d]_n / batch
  // Note: we divide by batch only (not batch*L_out) to match Dense's /m where m=batch,
  // and db already handles L_out averaging. This matches LLD §6.2: db = rowMean over batch*L_out.
  for (int oc = 0; oc < outCh; ++oc) {
    for (int ic = 0; ic < inCh; ++ic) {
      for (int k = 0; k < K; ++k) {
        double sum = 0.0;
        int w_col = ic * K + k;
        for (int n = 0; n < batch; ++n) {
          for (int ol = 0; ol < L_out; ++ol) {
            int z_row = oc * L_out + ol;
            double dz = dZ(z_row, n);
            int in_pos = ol * s - p + k * d;
            double xval = 0.0;
            if (in_pos >= 0 && in_pos < L_in) {
              int x_row = ic * L_in + in_pos;
              xval = input(x_row, n);
            }
            sum += dz * xval;
          }
        }
        dW(oc, w_col) = sum / static_cast<double>(batch);
      }
    }
  }

  return {dW, db};
}

void Conv1D::update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
                    double lr) noexcept {
  W_ -= lr * dW;
  b_ -= lr * db;
}

} // namespace FlexNN::Layers
