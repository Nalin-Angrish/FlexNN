/**
 * @file activation_test.cpp
 * @brief Tests for FlexNN::Activations (to_string, try_parse, forward/backward).
 *
 * Covers:
 * - to_string/try_parse round-trip for all 6 Activation values (+ aliases)
 * - Per-activation forward vs naive scalar
 * - Backward vs finite-difference (eps=1e-5) for scalar loss
 * - Softmax stability with large logits and column-wise semantics
 */

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "activations/Activation.hpp"
#include "activations/detail.hpp"

using namespace FlexNN::Activations;
using namespace FlexNN::Activations::detail;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Finite-difference gradient w.r.t. Z for a scalar loss `L = sum(forward(Z))`.
// Used to verify backward() for elementwise activations where dA = 1.
static Eigen::MatrixXd finite_diff_grad(Activation act, const Eigen::MatrixXd& Z,
                                        const ActivationParameters& params = {},
                                        double eps = 1e-5) {
  Eigen::MatrixXd grad(Z.rows(), Z.cols());
  for (Eigen::Index r = 0; r < Z.rows(); ++r) {
    for (Eigen::Index c = 0; c < Z.cols(); ++c) {
      Eigen::MatrixXd Zp = Z, Zm = Z;
      Zp(r, c) += eps;
      Zm(r, c) -= eps;
      double fp = forward(act, Zp, params).sum();
      double fm = forward(act, Zm, params).sum();
      grad(r, c) = (fp - fm) / (2 * eps);
    }
  }
  return grad;
}

// ---------------------------------------------------------------------------
// to_string / try_parse
// ---------------------------------------------------------------------------

TEST(ActivationString, ToStringExhaustive) {
  EXPECT_STREQ(to_string(Activation::None), "none");
  EXPECT_STREQ(to_string(Activation::ReLU), "relu");
  EXPECT_STREQ(to_string(Activation::LeakyReLU), "leaky_relu");
  EXPECT_STREQ(to_string(Activation::Sigmoid), "sigmoid");
  EXPECT_STREQ(to_string(Activation::Tanh), "tanh");
  EXPECT_STREQ(to_string(Activation::Softmax), "softmax");
}

TEST(ActivationString, TryParseRoundTrip) {
  for (Activation a : {Activation::None, Activation::ReLU, Activation::LeakyReLU,
                       Activation::Sigmoid, Activation::Tanh, Activation::Softmax}) {
    const char* s = to_string(a);
    Activation out;
    ASSERT_TRUE(try_parse(s, out)) << "failed to parse " << s;
    EXPECT_EQ(out, a) << "round-trip mismatch for " << s;
  }
}

TEST(ActivationString, TryParseCaseInsensitiveAndWhitespace) {
  Activation out;
  EXPECT_TRUE(try_parse("ReLU", out));
  EXPECT_EQ(out, Activation::ReLU);
  EXPECT_TRUE(try_parse("  relu  ", out));
  EXPECT_EQ(out, Activation::ReLU);
  EXPECT_TRUE(try_parse("LEAKY_RELU", out));
  EXPECT_EQ(out, Activation::LeakyReLU);
  EXPECT_TRUE(try_parse("leakyrelu", out));
  EXPECT_EQ(out, Activation::LeakyReLU);
  EXPECT_TRUE(try_parse("Leaky-Relu", out));
  EXPECT_EQ(out, Activation::LeakyReLU);
  EXPECT_TRUE(try_parse("SIGMOID", out));
  EXPECT_EQ(out, Activation::Sigmoid);
  EXPECT_TRUE(try_parse("Tanh", out));
  EXPECT_EQ(out, Activation::Tanh);
  EXPECT_TRUE(try_parse("  Softmax ", out));
  EXPECT_EQ(out, Activation::Softmax);
  EXPECT_TRUE(try_parse("NONE", out));
  EXPECT_EQ(out, Activation::None);
  EXPECT_TRUE(try_parse("linear", out)); // alias for None
  EXPECT_EQ(out, Activation::None);
}

TEST(ActivationString, TryParseUnknown) {
  Activation out = Activation::ReLU;
  EXPECT_FALSE(try_parse("", out));
  EXPECT_FALSE(try_parse("unknown", out));
  EXPECT_FALSE(try_parse("relu!", out));
  // out should be unchanged on failure (still ReLU from init)
  EXPECT_EQ(out, Activation::ReLU);
  EXPECT_FALSE(try_parse("   ", out));
}

// ---------------------------------------------------------------------------
// Forward correctness (vs naive)
// ---------------------------------------------------------------------------

TEST(ActivationForward, NoneIsIdentity) {
  Eigen::MatrixXd Z(2, 3);
  Z << 1, -2, 3, -4, 5, -6;
  EXPECT_EQ(forward(Activation::None, Z), Z);
}

TEST(ActivationForward, ReLU) {
  Eigen::MatrixXd Z(2, 2);
  Z << -1, 2, 0, -0.5;
  Eigen::MatrixXd A = forward(Activation::ReLU, Z);
  Eigen::MatrixXd expected(2, 2);
  expected << 0, 2, 0, 0;
  EXPECT_EQ(A, expected);
}

TEST(ActivationForward, LeakyReLU) {
  Eigen::MatrixXd Z(1, 3);
  Z << -2, 0, 2;
  Eigen::MatrixXd A = forward(Activation::LeakyReLU, Z);
  // 0 is on the leak side (Z<=0) per leaky_relu_backward definition (<=0 * alpha)
  // Forward at 0 is 0 as well (max branch is >0)
  EXPECT_DOUBLE_EQ(A(0, 0), -0.02); // -2 * 0.01
  EXPECT_DOUBLE_EQ(A(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(A(0, 2), 2.0);
}

TEST(ActivationForward, LeakyReLUCustomAlpha) {
  Eigen::MatrixXd Z(1, 3);
  Z << -2, 0, 2;
  ActivationParameters p;
  p.leakyAlpha = 0.2;
  Eigen::MatrixXd A = forward(Activation::LeakyReLU, Z, p);
  EXPECT_DOUBLE_EQ(A(0, 0), -0.4); // -2 * 0.2
  EXPECT_DOUBLE_EQ(A(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(A(0, 2), 2.0);
  // Direct helper with explicit alpha
  Eigen::MatrixXd A2 = leaky_relu_forward(Z, 0.2);
  EXPECT_EQ(A, A2);
}

TEST(ActivationParameters, DefaultsAndEquality) {
  ActivationParameters p1;
  EXPECT_DOUBLE_EQ(p1.leakyAlpha, 0.01);
  EXPECT_DOUBLE_EQ(p1.leakyAlpha, kDefaultLeakyAlpha);
  ActivationParameters p2;
  p2.leakyAlpha = 0.2;
  EXPECT_NE(p1, p2);
  ActivationParameters p3;
  p3.leakyAlpha = 0.2;
  EXPECT_EQ(p2, p3);
}

TEST(ActivationForward, SigmoidClamped) {
  Eigen::MatrixXd Z(1, 3);
  Z << -100, 0, 100; // extreme values should be clamped to [-15,15] before exp
  Eigen::MatrixXd A = forward(Activation::Sigmoid, Z);
  // sigmoid(0)=0.5, sigmoid(15) ~ 0.999999, sigmoid(-15) ~ 3e-7
  EXPECT_NEAR(A(0, 1), 0.5, 1e-9);
  EXPECT_GT(A(0, 2), 0.9999);
  EXPECT_LT(A(0, 0), 0.0001);
  // No NaN/Inf despite large input
  EXPECT_TRUE(A.allFinite());
}

TEST(ActivationForward, Tanh) {
  Eigen::MatrixXd Z(1, 3);
  Z << -1, 0, 1;
  Eigen::MatrixXd A = forward(Activation::Tanh, Z);
  EXPECT_NEAR(A(0, 0), std::tanh(-1.0), 1e-12);
  EXPECT_NEAR(A(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(A(0, 2), std::tanh(1.0), 1e-12);
}

TEST(ActivationForward, SoftmaxStableAndColumnWise) {
  Eigen::MatrixXd Z(2, 2);
  // Column 0 has large identical logits -> softmax ~ [0.5,0.5] and stable (no overflow)
  // Column 1 is [0,1] -> softmax ~ [0.2689, 0.7310]
  Z << 1000, 0,
       1000, 1;
  Eigen::MatrixXd A = forward(Activation::Softmax, Z);
  EXPECT_TRUE(A.allFinite()) << "softmax overflowed";
  // Column 0: max=1000, exp(0)=1, sum=2
  EXPECT_NEAR(A(0, 0), 0.5, 1e-9);
  EXPECT_NEAR(A(1, 0), 0.5, 1e-9);
  // Column 1
  double s0 = std::exp(0.0) / (std::exp(0.0) + std::exp(1.0));
  double s1 = std::exp(1.0) / (std::exp(0.0) + std::exp(1.0));
  EXPECT_NEAR(A(0, 1), s0, 1e-9);
  EXPECT_NEAR(A(1, 1), s1, 1e-9);
  // Columns sum to 1
  EXPECT_NEAR(A.col(0).sum(), 1.0, 1e-9);
  EXPECT_NEAR(A.col(1).sum(), 1.0, 1e-9);
}

// Direct per-header helpers
TEST(ActivationForward, SoftmaxDirectHelper) {
  Eigen::MatrixXd Z(2, 1);
  Z << 1, 2;
  Eigen::MatrixXd A = softmax_forward(Z);
  Eigen::MatrixXd B = forward(Activation::Softmax, Z);
  EXPECT_EQ(A, B);
}

// ---------------------------------------------------------------------------
// Backward vs finite difference (elementwise activations)
// ---------------------------------------------------------------------------

TEST(ActivationBackward, None) {
  Eigen::MatrixXd Z = Eigen::MatrixXd::Random(3, 4);
  Eigen::MatrixXd A = forward(Activation::None, Z);
  Eigen::MatrixXd dA = Eigen::MatrixXd::Ones(3, 4);
  Eigen::MatrixXd dZ = backward(Activation::None, dA, Z, A);
  EXPECT_EQ(dZ, dA);
}

TEST(ActivationBackward, ReLUFiniteDiff) {
  // Avoid Z==0 boundary for stable finite diff
  Eigen::MatrixXd Z(2, 3);
  Z << 0.5, -0.5, 1.5,
       -1.5, 2.0, -0.1;
  Eigen::MatrixXd A = forward(Activation::ReLU, Z);
  Eigen::MatrixXd dA = Eigen::MatrixXd::Ones(2, 3);
  Eigen::MatrixXd dZ = backward(Activation::ReLU, dA, Z, A);
  Eigen::MatrixXd fd = finite_diff_grad(Activation::ReLU, Z);
  for (Eigen::Index r = 0; r < Z.rows(); ++r) {
    for (Eigen::Index c = 0; c < Z.cols(); ++c) {
      EXPECT_NEAR(dZ(r, c), fd(r, c), 1e-4) << "at (" << r << "," << c << ") Z=" << Z(r,c);
    }
  }
}

TEST(ActivationBackward, LeakyReLUFiniteDiff) {
  Eigen::MatrixXd Z(2, 3);
  Z << 0.5, -0.5, 1.5,
       -1.5, 2.0, -0.1;
  Eigen::MatrixXd A = forward(Activation::LeakyReLU, Z);
  Eigen::MatrixXd dA = Eigen::MatrixXd::Ones(2, 3);
  Eigen::MatrixXd dZ = backward(Activation::LeakyReLU, dA, Z, A);
  Eigen::MatrixXd fd = finite_diff_grad(Activation::LeakyReLU, Z);
  EXPECT_TRUE((dZ - fd).cwiseAbs().maxCoeff() < 1e-4);
}

TEST(ActivationBackward, LeakyReLUCustomAlphaFiniteDiff) {
  Eigen::MatrixXd Z(2, 3);
  Z << 0.5, -0.5, 1.5,
       -1.5, 2.0, -0.1;
  ActivationParameters p;
  p.leakyAlpha = 0.2;
  Eigen::MatrixXd A = forward(Activation::LeakyReLU, Z, p);
  Eigen::MatrixXd dA = Eigen::MatrixXd::Ones(2, 3);
  Eigen::MatrixXd dZ = backward(Activation::LeakyReLU, dA, Z, A, p);
  Eigen::MatrixXd fd = finite_diff_grad(Activation::LeakyReLU, Z, p);
  EXPECT_TRUE((dZ - fd).cwiseAbs().maxCoeff() < 1e-4);
  // Direct helper
  Eigen::MatrixXd dZ2 = leaky_relu_backward(dA, Z, A, 0.2);
  EXPECT_EQ(dZ, dZ2);
}

TEST(ActivationBackward, SigmoidFiniteDiff) {
  Eigen::MatrixXd Z(2, 3);
  Z << -1, 0, 1,
       0.5, -0.5, 2;
  Eigen::MatrixXd A = forward(Activation::Sigmoid, Z);
  Eigen::MatrixXd dA = Eigen::MatrixXd::Ones(2, 3);
  Eigen::MatrixXd dZ = backward(Activation::Sigmoid, dA, Z, A);
  Eigen::MatrixXd fd = finite_diff_grad(Activation::Sigmoid, Z);
  EXPECT_TRUE((dZ - fd).cwiseAbs().maxCoeff() < 1e-4);
}

TEST(ActivationBackward, TanhFiniteDiff) {
  Eigen::MatrixXd Z(2, 3);
  Z << -1, 0, 1,
       0.5, -0.5, 0.2;
  Eigen::MatrixXd A = forward(Activation::Tanh, Z);
  Eigen::MatrixXd dA = Eigen::MatrixXd::Ones(2, 3);
  Eigen::MatrixXd dZ = backward(Activation::Tanh, dA, Z, A);
  Eigen::MatrixXd fd = finite_diff_grad(Activation::Tanh, Z);
  EXPECT_TRUE((dZ - fd).cwiseAbs().maxCoeff() < 1e-4);
}

// Softmax hidden backward: J * upstream vs finite diff through softmax sum loss
TEST(ActivationBackward, SoftmaxHiddenFiniteDiff) {
  Eigen::MatrixXd Z(3, 2);
  Z << 0.5, -0.2,
       1.0,  0.3,
      -0.5,  0.1;
  Eigen::MatrixXd s = softmax_forward(Z);
  // upstream = dL/dA where L = sum(s) ??? But sum(s)=batch -> not useful.
  // Instead test Jacobian action: finite diff of `sum(upstream ⊙ s)` w.r.t. Z
  // Equivalent to: define loss L = sum_c (up_c^T * s_c), then dZ = J^T * up = J*up (J symmetric)
  Eigen::MatrixXd up = Eigen::MatrixXd::Random(3, 2);
  Eigen::MatrixXd dZ = softmax_backward_hidden(s, up);
  Eigen::MatrixXd fd(3, 2);
  double eps = 1e-5;
  for (Eigen::Index r = 0; r < Z.rows(); ++r) {
    for (Eigen::Index c = 0; c < Z.cols(); ++c) {
      Eigen::MatrixXd Zp = Z, Zm = Z;
      Zp(r, c) += eps;
      Zm(r, c) -= eps;
      Eigen::MatrixXd sp = softmax_forward(Zp);
      Eigen::MatrixXd sm = softmax_forward(Zm);
      double fp = (up.array() * sp.array()).sum();
      double fm = (up.array() * sm.array()).sum();
      fd(r, c) = (fp - fm) / (2 * eps);
    }
  }
  EXPECT_TRUE((dZ - fd).cwiseAbs().maxCoeff() < 1e-4);
}
