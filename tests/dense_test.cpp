/**
 * @file dense_test.cpp
 * @brief Tests for FlexNN::Layers::Dense vs legacy and finite-diff.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "layers/Dense.hpp"
#include "activations/Activation.hpp"

using namespace FlexNN::Layers;
using namespace FlexNN::Activations;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Eigen::MatrixXd naive_dense_forward(const Eigen::MatrixXd& W,
                                           const Eigen::VectorXd& b,
                                           const Eigen::MatrixXd& X,
                                           Activation act) {
  Eigen::MatrixXd Z = (W * X).colwise() + b;
  return detail::forward(act, Z);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

TEST(DenseForward, MatchesNaive) {
  Dense d(4, 3, Activation::ReLU);
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(4, 5);
  auto [Z, A] = d.forward(X);
  Eigen::MatrixXd Znaive = (d.weights() * X).colwise() + d.biases();
  Eigen::MatrixXd Anaive = naive_dense_forward(d.weights(), d.biases(), X, d.activation());
  EXPECT_TRUE(Z.isApprox(Znaive, 1e-9));
  EXPECT_TRUE(A.isApprox(Anaive, 1e-9));
  // Check dimensions [out × batch]
  EXPECT_EQ(Z.rows(), 3);
  EXPECT_EQ(Z.cols(), 5);
  EXPECT_EQ(A.rows(), 3);
  EXPECT_EQ(A.cols(), 5);
}

TEST(DenseForward, DifferentActivations) {
  for (Activation act :
       {Activation::None, Activation::ReLU, Activation::LeakyReLU,
        Activation::Sigmoid, Activation::Tanh}) {
    Dense d(2, 2, act);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 4);
    auto [Z, A] = d.forward(X);
    Eigen::MatrixXd expected = detail::forward(act, Z);
    EXPECT_TRUE(A.isApprox(expected, 1e-12)) << "failed for act " << to_string(act);
  }
}

// Softmax forward is column-wise and sums to 1
TEST(DenseForward, Softmax) {
  Dense d(3, 3, Activation::Softmax);
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(3, 4);
  auto [Z, A] = d.forward(X);
  for (Eigen::Index c = 0; c < A.cols(); ++c) {
    EXPECT_NEAR(A.col(c).sum(), 1.0, 1e-9);
    EXPECT_TRUE((A.col(c).array() >= 0).all());
  }
}

// ---------------------------------------------------------------------------
// Backward vs finite diff
// ---------------------------------------------------------------------------

TEST(DenseBackward, FiniteDiff) {
  // Test hidden Dense backward: dZ = f'(Z) ⊙ (W_next^T * dZ_next)
  // We use a small network fragment: Dense(2->3) followed by Dummy nextW [4×3]
  for (Activation act :
       {Activation::ReLU, Activation::LeakyReLU, Activation::Sigmoid,
        Activation::Tanh, Activation::None}) {
    Dense cur(2, 3, act);
    // Avoid ReLU boundary at 0 for stable FD
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 2);
    auto [Z, A] = cur.forward(X);
    // Ensure Z not exactly 0 for ReLU
    if (act == Activation::ReLU) {
      for (Eigen::Index i = 0; i < Z.size(); ++i) {
        if (std::abs(Z(i)) < 1e-6) Z(i) = 0.5;
      }
      // Recompute A after tweak
      A = detail::forward(act, Z);
    }

    Eigen::MatrixXd nextW = Eigen::MatrixXd::Random(4, 3);
    Eigen::MatrixXd nextdZ = Eigen::MatrixXd::Random(4, 2);
    Eigen::MatrixXd dZ = cur.backward(nextW, nextdZ, Z);

    // Finite diff: dZ = ∂L/∂Z where L = sum(A_next) and A_next = forward(Z)
    // But cur.backward expects upstream = W_next^T * dZ_next, so we finite-diff
    // L = sum( detail::forward(act, Z) * up_weight )? Simpler: directly
    // finite-diff the scalar loss L = sum( up^T * A ) where up = W_next^T * dZ_next
    // For our test, set up = nextW^T * nextdZ, and L = sum(up ⊙ A)
    Eigen::MatrixXd up = nextW.transpose() * nextdZ;
    auto loss = [&](const Eigen::MatrixXd& Zv) {
      Eigen::MatrixXd Av = detail::forward(act, Zv);
      return (up.array() * Av.array()).sum();
    };
    double eps = 1e-5;
    Eigen::MatrixXd fd(3, 2);
    for (Eigen::Index r = 0; r < 3; ++r) {
      for (Eigen::Index c = 0; c < 2; ++c) {
        Eigen::MatrixXd Zp = Z, Zm = Z;
        Zp(r, c) += eps;
        Zm(r, c) -= eps;
        double fp = loss(Zp);
        double fm = loss(Zm);
        fd(r, c) = (fp - fm) / (2 * eps);
      }
    }
    double maxErr = (dZ - fd).cwiseAbs().maxCoeff();
    EXPECT_LT(maxErr, 1e-4) << "act " << to_string(act) << " maxErr " << maxErr;
  }
}

TEST(Dense, StringShim) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  Dense d1(4, 2, std::string("relu"));
#pragma GCC diagnostic pop
  Dense d2(4, 2, Activation::ReLU);
  EXPECT_EQ(d1.activation(), d2.activation());
  EXPECT_EQ(d1.activation(), Activation::ReLU);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  Dense d3(4, 2, std::string("UNKNOWN_ACTIVATION"));
#pragma GCC diagnostic pop
  EXPECT_EQ(d3.activation(), Activation::None);
}

TEST(Dense, ParamsAccessor) {
  Dense d(DenseParams{5, 7}, Activation::Tanh);
  EXPECT_EQ(d.params().inputSize, 5);
  EXPECT_EQ(d.params().outputSize, 7);
  EXPECT_EQ(d.activation(), Activation::Tanh);
  EXPECT_EQ(d.type(), LayerType::Dense);
  EXPECT_EQ(d.weights().rows(), 7);
  EXPECT_EQ(d.weights().cols(), 5);
  EXPECT_EQ(d.biases().size(), 7);
}
