/**
 * @file Conv1D.cpp
 * @brief Implements FlexNN::Layers::Conv1D forward/backward/propagate.
 *
 * Forward uses im2col + single GEMM (`Zflat = W * Col`) so Eigen can
 * vectorize/thread the hot math; padding is zero-fill at im2col time so
 * the inner loop has no branch. Propagate (col2im) and grad share the
 * same helper. Results match the naive loops to ~1e-12.
 * The logic matches LLD_FLEXNN §6.2 and is verified by `tests/conv1d_test.cpp`
 * via brute force and finite differences.
 */

#include "layers/Conv1D.hpp"

#include <cassert>
#include <cmath>

#include "activations/detail.hpp"

namespace FlexNN::Layers {

namespace {
// Build im2col matrix [inCh*K × L_out*batch]: each column is one receptive
// field, zero-filled where the window overlaps padding (so the GEMM hot
// loop has no branch). Row layout: row = ic*K + k.
Eigen::MatrixXd im2col(const Eigen::MatrixXd& input, int inCh, int L_in,
                       int K, int s, int p, int d, int L_out, int batch) {
  Eigen::MatrixXd col(inCh * K, L_out * batch);
  col.setZero();
#pragma omp parallel for collapse(2) if (batch * L_out > 64)
  for (int n = 0; n < batch; ++n) {
    for (int ol = 0; ol < L_out; ++ol) {
      int j = n * L_out + ol;
      for (int ic = 0; ic < inCh; ++ic) {
        int base = ic * L_in;
        int row = ic * K;
        for (int k = 0; k < K; ++k) {
          int in_pos = ol * s - p + k * d;
          if (in_pos >= 0 && in_pos < L_in) {
            col(row + k, j) = input(base + in_pos, n);
          }
        }
      }
    }
  }
  return col;
}
}  // namespace

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

  // im2col + GEMM: one BLAS call replaces the
  // batch·outCh·L_out·inCh·K scalar nest (and its padding branch).
  Eigen::MatrixXd col = im2col(input, inCh, L_in, K, s, p, d, L_out, batch);
  Eigen::MatrixXd Zflat = W_ * col;  // [outCh × L_out*batch]
  Zflat.colwise() += b_;

  // Reshape [outCh × L_out*batch] -> [outCh*L_out × batch]
  // (row = oc*L_out + ol, col = n). Cheap vs the GEMM above.
#pragma omp parallel for collapse(2) if (batch * outCh > 32)
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < outCh; ++oc) {
      for (int ol = 0; ol < L_out; ++ol) {
        Z(oc * L_out + ol, n) = Zflat(oc, n * L_out + ol);
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

  // Reshape dZ [outCh*L_out × batch] -> [outCh × L_out*batch], then one GEMM
  // gives all transposed-window contributions at once; scatter-add (col2im)
  // folds overlaps (stride < K) and drops padding.
  Eigen::MatrixXd dZflat(outCh, L_out * batch);
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < outCh; ++oc) {
      for (int ol = 0; ol < L_out; ++ol) {
        dZflat(oc, n * L_out + ol) = dZ(oc * L_out + ol, n);
      }
    }
  }
  Eigen::MatrixXd up = W_.transpose() * dZflat;  // [inCh*K × L_out*batch]

#pragma omp parallel for if (batch > 4)
  for (int n = 0; n < batch; ++n) {
    for (int ol = 0; ol < L_out; ++ol) {
      int j = n * L_out + ol;
      for (int ic = 0; ic < inCh; ++ic) {
        int base = ic * L_in;
        int row = ic * K;
        for (int k = 0; k < K; ++k) {
          int in_pos = ol * s - p + k * d;
          if (in_pos >= 0 && in_pos < L_in) {
            dX(base + in_pos, n) += up(row + k, j);
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

  // dW via im2col: dW = dZflat * Col^T / batch, where Col is the same
  // receptive-field matrix as forward. One GEMM replaces the
  // oc·ic·K·batch·L_out nest. (db above already averages over batch*L_out,
  // matching LLD §6.2; dW divides by batch only to match Dense's /m.)
  Eigen::MatrixXd col = im2col(input, inCh, L_in, K, s, p, d, L_out, batch);
  Eigen::MatrixXd dZflat(outCh, L_out * batch);
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < outCh; ++oc) {
      for (int ol = 0; ol < L_out; ++ol) {
        dZflat(oc, n * L_out + ol) = dZ(oc * L_out + ol, n);
      }
    }
  }
  dW = (dZflat * col.transpose()) / static_cast<double>(batch);

  return {dW, db};
}

void Conv1D::update(const Eigen::MatrixXd& dW, const Eigen::VectorXd& db,
                    double lr) noexcept {
  W_ -= lr * dW;
  b_ -= lr * db;
}

} // namespace FlexNN::Layers
