/**
 * @file conv1d_test.cpp
 * @brief Tests for FlexNN::Layers::Conv1D (forward vs brute, backward finitediff).
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/Conv1D.hpp"
#include "layers/Layer.hpp"
#include "FlexNN.h"
#include "activations/Activation.hpp"

using namespace FlexNN::Layers;
using namespace FlexNN::Activations;

// Brute-force Conv1D forward (same as implementation but explicit for test)
static Eigen::MatrixXd brute_conv1d_forward(const Conv1D& conv,
                                            const Eigen::MatrixXd& X) {
  auto params = conv.params();
  int inCh = params.inChannels;
  int outCh = params.outChannels;
  int K = params.kernelSize;
  int s = params.stride;
  int p = params.padding;
  int d = params.dilation;
  int L_in = static_cast<int>(X.rows() / inCh);
  int L_out = (L_in + 2 * p - d * (K - 1) - 1) / s + 1;
  int batch = static_cast<int>(X.cols());
  Eigen::MatrixXd Z(outCh * L_out, batch);
  Z.setZero();
  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < outCh; ++oc) {
      for (int ol = 0; ol < L_out; ++ol) {
        double sum = conv.biases()(oc);
        for (int ic = 0; ic < inCh; ++ic) {
          for (int k = 0; k < K; ++k) {
            int in_pos = ol * s - p + k * d;
            double xval = 0.0;
            if (in_pos >= 0 && in_pos < L_in) {
              int row = ic * L_in + in_pos;
              xval = X(row, n);
            }
            int col = ic * K + k;
            sum += conv.weights()(oc, col) * xval;
          }
        }
        int z_row = oc * L_out + ol;
        Z(z_row, n) = sum;
      }
    }
  }
  // No activation in brute (we test conv linear part); caller adds act
  return Z;
}

TEST(Conv1DForward, MatchesBruteSmall) {
  // Small sweep: C_in 1..2, L 4..8, K 1..3, stride 1..2, pad 0..1
  for (int Cin : {1, 2}) {
    for (int L : {4, 6}) {
      for (int K : {1, 3}) {
        for (int stride : {1, 2}) {
          for (int pad : {0, 1}) {
            Conv1DParams p{Cin, 2, K, stride, pad, 1};
            int L_out = (L + 2 * pad - (K - 1) - 1) / stride + 1;
            if (L_out < 1) continue;
            Conv1D conv(p, Activation::None);
            Eigen::MatrixXd X(Cin * L, 2);
            X.setRandom();
            auto [Z, A] = conv.forward(X);
            Eigen::MatrixXd Zbrute = brute_conv1d_forward(conv, X);
            EXPECT_TRUE(Z.isApprox(Zbrute, 1e-9))
                << " Cin=" << Cin << " L=" << L << " K=" << K << " s=" << stride << " p=" << pad;
            // None activation => A == Z
            EXPECT_TRUE(A.isApprox(Z, 1e-9));
          }
        }
      }
    }
  }
}

TEST(Conv1DForward, WithReLU) {
  Conv1D conv(Conv1DParams{1, 1, 3, 1, 1, 1}, Activation::ReLU);
  Eigen::MatrixXd X(4, 1);
  X << -2, -1, 1, 2; // L_in=4, C_in=1
  auto [Z, A] = conv.forward(X);
  // Check A = max(0,Z) elementwise
  for (Eigen::Index i = 0; i < Z.size(); ++i) {
    double expected = std::max(0.0, Z(i));
    EXPECT_DOUBLE_EQ(A(i), expected);
  }
}

TEST(Conv1DForward, Dilation) {
  Conv1D conv(Conv1DParams{1, 1, 2, 1, 0, 2}, Activation::None);
  Eigen::MatrixXd X(5, 1); // L_in=5
  X << 1, 2, 3, 4, 5;
  auto [Z, A] = conv.forward(X);
  // Manual: L_out = (5 -2*(1) -1)/1 +1 =3
  EXPECT_EQ(Z.rows(), 3);
  // We just check it runs and matches brute
  Eigen::MatrixXd Zbrute = brute_conv1d_forward(conv, X);
  EXPECT_TRUE(Z.isApprox(Zbrute, 1e-9));
}

TEST(Conv1DGrad, FiniteDiff) {
  Conv1D conv(Conv1DParams{2, 3, 3, 1, 1, 1}, Activation::None);
  int Cin = 2, L_in = 6;
  int batch = 2;
  Eigen::MatrixXd X(Cin * L_in, batch);
  X.setRandom();
  auto [Z, A] = conv.forward(X);
  // Use dZ = ones to test grad. Note: conv.grad divides by batch for dW and by batch*L_out for db
  // (see LLD §6.2), while finite diff with loss = sum(dZ ⊙ A) gives unscaled sum.
  // So we compare dW*batch vs sum and db*batch*L_out vs sum.
  Eigen::MatrixXd dZ = Eigen::MatrixXd::Ones(Z.rows(), Z.cols());
  auto [dW, db] = conv.grad(dZ, X);
  int L_out = static_cast<int>(Z.rows() / conv.params().outChannels);

  // Finite diff for dW
  double eps = 1e-4;
  Eigen::MatrixXd dW_fd(dW.rows(), dW.cols());
  dW_fd.setZero();
  for (int r = 0; r < dW.rows(); ++r) {
    for (int c = 0; c < dW.cols(); ++c) {
      Conv1D conv_p = conv;
      Conv1D conv_m = conv;
      Eigen::MatrixXd W = conv.weights();
      W(r, c) += eps;
      conv_p.setWeights(W);
      W(r, c) -= 2 * eps;
      conv_m.setWeights(W);
      auto [Zp, Ap] = conv_p.forward(X);
      auto [Zm, Am] = conv_m.forward(X);
      double fp = (dZ.array() * Ap.array()).sum(); // loss = sum(dZ ⊙ A) — unscaled
      double fm = (dZ.array() * Am.array()).sum();
      dW_fd(r, c) = (fp - fm) / (2 * eps);
    }
  }
  // dW is sum / batch, so scale back for comparison
  double maxErrW = (dW * static_cast<double>(batch) - dW_fd).cwiseAbs().maxCoeff();
  EXPECT_LT(maxErrW, 1e-3) << "dW maxErr " << maxErrW;

  // Finite diff for db (bias) — db is mean over batch*L_out, sum is *batch*L_out
  Eigen::VectorXd db_fd(db.size());
  db_fd.setZero();
  for (int oc = 0; oc < db.size(); ++oc) {
    Conv1D conv_p = conv;
    Conv1D conv_m = conv;
    Eigen::VectorXd b = conv.biases();
    b(oc) += eps;
    conv_p.setBiases(b);
    b(oc) -= 2 * eps;
    conv_m.setBiases(b);
    auto [Zp, Ap] = conv_p.forward(X);
    auto [Zm, Am] = conv_m.forward(X);
    double fp = (dZ.array() * Ap.array()).sum();
    double fm = (dZ.array() * Am.array()).sum();
    db_fd(oc) = (fp - fm) / (2 * eps);
  }
  double maxErrB = (db * static_cast<double>(batch * L_out) - db_fd).cwiseAbs().maxCoeff();
  EXPECT_LT(maxErrB, 1e-3) << "db maxErr " << maxErrB;
}

TEST(Conv1DPropagate, FiniteDiff) {
  Conv1D conv(Conv1DParams{1, 2, 3, 1, 1, 1}, Activation::None);
  int Cin = 1, L_in = 5;
  int batch = 1;
  Eigen::MatrixXd X(Cin * L_in, batch);
  X.setRandom();
  auto [Z, A] = conv.forward(X);
  Eigen::MatrixXd dZ = Eigen::MatrixXd::Random(Z.rows(), Z.cols());
  Eigen::MatrixXd dX = conv.propagate(dZ);

  // Finite diff: dX = ∂L/∂X where L = sum(dZ ⊙ A)
  Eigen::MatrixXd dX_fd(Cin * L_in, batch);
  double eps = 1e-5;
  for (int r = 0; r < X.rows(); ++r) {
    for (int c = 0; c < X.cols(); ++c) {
      Eigen::MatrixXd Xp = X, Xm = X;
      Xp(r, c) += eps;
      Xm(r, c) -= eps;
      auto [Zp, Ap] = conv.forward(Xp);
      auto [Zm, Am] = conv.forward(Xm);
      double fp = (dZ.array() * Ap.array()).sum();
      double fm = (dZ.array() * Am.array()).sum();
      dX_fd(r, c) = (fp - fm) / (2 * eps);
    }
  }
  EXPECT_NEAR((dX - dX_fd).norm(), 0.0, 1e-3);
}

TEST(Conv1DVariant, InLayer) {
  Conv1D conv(Conv1DParams{1, 2, 3, 1, 1, 1}, Activation::ReLU);
  Layer l(conv);
  EXPECT_EQ(l.type(), LayerType::Conv1D);
  EXPECT_EQ(l.activation(), Activation::ReLU);
  ASSERT_NE(l.asConv1D(), nullptr);
  EXPECT_EQ(l.asConv1D()->params().inChannels, 1);
}

TEST(Conv1DStringShim, Deprecated) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  Conv1D conv(Conv1DParams{1, 2, 3, 1, 1, 1}, std::string("relu"));
#pragma GCC diagnostic pop
  EXPECT_EQ(conv.activation(), Activation::ReLU);
}

TEST(Conv1DNetwork, DenseAfterConv) {
  // Conv1D -> Dense -> Softmax stack should train without NaN
  // Input: C=2, L=8 => in dim 16
  Eigen::MatrixXd X(16, 4);
  X.setRandom();
  Eigen::VectorXd Y(4);
  Y << 0, 1, 0, 1;

  FlexNN::NeuralNetwork net(std::vector<FlexNN::Layers::Layer>{
      Conv1D(Conv1DParams{2, 3, 3, 1, 1, 1}, Activation::ReLU), // 2*8 -> 3*8
      FlexNN::Layers::Dense(3 * 8, 2, Activation::Softmax),
  });
  Eigen::MatrixXd pred_before = net.predict(X);
  EXPECT_TRUE(pred_before.allFinite());
  net.train(X, Y, 0.1, 5);
  Eigen::MatrixXd pred_after = net.predict(X);
  EXPECT_TRUE(pred_after.allFinite());
}
