/**
 * @file batchnorm_test.cpp
 * @brief Tests for BatchNorm1D forward, running stats, and ModelIO.
 */

#include <gtest/gtest.h>

#include "layers/BatchNorm1D.hpp"
#include "layers/Layer.hpp"
#include "FlexNN.h"
#include "ModelIO.hpp"
#include "activations/Activation.hpp"

using namespace FlexNN::Layers;
using namespace FlexNN::Activations;

TEST(BatchNormForward, BatchVsRunning) {
  BatchNormParams p{3};
  BatchNorm1D bn(p, Activation::None);
  Eigen::MatrixXd X(3, 4);
  X << 1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10, 11, 12;
  // Training forward uses batch stats
  auto [Z_train, A_train] = bn.forward(X, true);
  EXPECT_EQ(Z_train.rows(), 3);
  EXPECT_EQ(Z_train.cols(), 4);
  // Eval uses running stats (initially mean 0 var 1)
  auto [Z_eval, A_eval] = bn.forward(X, false);
  EXPECT_TRUE(Z_eval.allFinite());
  // Training and eval differ because batch stats != running
  EXPECT_FALSE(Z_train.isApprox(Z_eval, 1e-6));
}

TEST(BatchNormForward, GammaBeta) {
  BatchNormParams p{2};
  BatchNorm1D bn(p, Activation::None);
  // Set gamma=2, beta=1
  Eigen::VectorXd gamma(2), beta(2);
  gamma << 2, 2;
  beta << 1, 1;
  bn.setGamma(gamma);
  bn.setBeta(beta);
  Eigen::MatrixXd X(2, 2);
  X << 0, 0,
       0, 0;
  // With X=0, batch mean 0 var 0 -> y = gamma*(0-0)/sqrt(eps) + beta = beta
  auto [Z, A] = bn.forward(X, true);
  // For X=0, normalized is 0, so y = beta =1
  EXPECT_NEAR(Z(0, 0), 1.0, 1e-6);
  EXPECT_NEAR(Z(1, 0), 1.0, 1e-6);
}

TEST(BatchNormVariant, InLayer) {
  BatchNorm1D bn(BatchNormParams{4}, Activation::None);
  Layer l(bn);
  EXPECT_EQ(l.type(), LayerType::BatchNorm1D);
  EXPECT_EQ(l.activation(), Activation::None);
  ASSERT_NE(l.asBatchNorm1D(), nullptr);
  EXPECT_EQ(l.asBatchNorm1D()->params().numFeatures, 4);
}

TEST(BatchNormModelIO, RoundTrip) {
  FlexNN::NeuralNetwork net(std::vector<FlexNN::Layers::Layer>{
      BatchNorm1D(BatchNormParams{3}, Activation::None),
      Dense(3, 2, Activation::Softmax),
  });
  const std::string path = "/tmp/flexnn_bn_roundtrip.bin";
  auto st = FlexNN::exportModel(net, path);
  ASSERT_TRUE(st.ok) << st.error;
  FlexNN::NeuralNetwork net2(std::vector<FlexNN::Layers::Layer>{Dense(1, 1)});
  auto st2 = FlexNN::importModel(net2, path);
  ASSERT_TRUE(st2.ok) << st2.error;
  ASSERT_EQ(net2.layers().size(), 2u);
  EXPECT_EQ(net2.layers()[0].type(), LayerType::BatchNorm1D);
  const auto* bn = net2.layers()[0].asBatchNorm1D();
  const auto* orig = net.layers()[0].asBatchNorm1D();
  ASSERT_NE(bn, nullptr);
  ASSERT_NE(orig, nullptr);
  EXPECT_TRUE(bn->gamma().isApprox(orig->gamma(), 1e-6));
  EXPECT_TRUE(bn->beta().isApprox(orig->beta(), 1e-6));
  std::remove(path.c_str());
}

TEST(BatchNormModelIO, ActivationMustBeNone) {
  FlexNN::NeuralNetwork net(std::vector<FlexNN::Layers::Layer>{
      BatchNorm1D(BatchNormParams{2}, Activation::ReLU), // invalid
  });
  auto st = FlexNN::exportModel(net, "/tmp/flexnn_bn_act.bin");
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.error.find("BatchNorm"), std::string::npos);
}
