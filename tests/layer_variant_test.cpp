/**
 * @file layer_variant_test.cpp
 * @brief Tests for FlexNN::Layers::Layer variant wrapper and NeuralNetwork migration.
 *
 * Checks:
 * - Layer variant holds Dense correctly, type()/activation() dispatch, forward/backward via visit
 * - NeuralNetwork with vector<Layers::Layer> trains on toy problem and forward interleaving
 * - Legacy vector<Layer> (string) still converts and trains (deprecated shim)
 */

#include <gtest/gtest.h>

#include "FlexNN.h"
#include "Layer.h"
#include "layers/Dense.hpp"
#include "layers/Layer.hpp"
#include "activations/Activation.hpp"

using namespace FlexNN;
using namespace FlexNN::Activations;

TEST(LayerVariant, HoldsDense) {
  Layers::Dense d(4, 3, Activation::ReLU);
  Layers::Layer l(d);
  EXPECT_EQ(l.type(), Layers::LayerType::Dense);
  EXPECT_EQ(l.activation(), Activation::ReLU);
  ASSERT_NE(l.asDense(), nullptr);
  EXPECT_EQ(l.asDense()->params().inputSize, 4);
  EXPECT_EQ(l.asDense()->params().outputSize, 3);
}

TEST(LayerVariant, ForwardDispatch) {
  Layers::Dense d(2, 2, Activation::None);
  // Set known weights: W = [[1,0],[0,1]] (identity), b = [0,0]
  Eigen::MatrixXd W = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
  d.setWeights(W);
  d.setBiases(b);
  Layers::Layer l(d);
  Eigen::MatrixXd X(2, 2);
  X << 1, 2,
       3, 4;
  auto [Z, A] = l.forward(X);
  // Z = W*X + b = X, A = Z since None
  EXPECT_TRUE(Z.isApprox(X, 1e-9));
  EXPECT_TRUE(A.isApprox(X, 1e-9));
  // Check dispatch via variant
  auto [Z2, A2] = d.forward(X);
  EXPECT_TRUE(Z.isApprox(Z2, 1e-9));
  EXPECT_TRUE(A.isApprox(A2, 1e-9));
}

TEST(LayerVariant, WeightsMatrix) {
  Layers::Dense d(3, 2, Activation::Tanh);
  Layers::Layer l(d);
  Eigen::MatrixXd w = l.weightsMatrix();
  EXPECT_EQ(w.rows(), 2);
  EXPECT_EQ(w.cols(), 3);
  EXPECT_TRUE(w.isApprox(d.weights(), 1e-9));
}

TEST(LayerVariant, UpdateDispatch) {
  Layers::Dense d(2, 2, Activation::ReLU);
  Layers::Layer l(d);
  Eigen::MatrixXd dW = Eigen::MatrixXd::Ones(2, 2);
  Eigen::VectorXd db = Eigen::VectorXd::Ones(2);
  Eigen::MatrixXd W_before = l.asDense()->weights();
  l.updateWeights(dW, db, 0.1);
  Eigen::MatrixXd W_after = l.asDense()->weights();
  EXPECT_TRUE((W_before - W_after).isApprox(0.1 * dW, 1e-9));
}

TEST(NeuralNetworkNewAPI, ForwardInterleaving) {
  // Network: Dense(2->3, ReLU) -> Dense(3->2, Softmax)
  NeuralNetwork net(std::vector<Layers::Layer>{
      Layers::Dense(2, 3, Activation::ReLU),
      Layers::Dense(3, 2, Activation::Softmax),
  });
  Eigen::MatrixXd X(2, 4);
  X.setRandom();
  Eigen::MatrixXd pred = net.predict(X);
  EXPECT_EQ(pred.rows(), 2);
  EXPECT_EQ(pred.cols(), 4);
  // Softmax columns sum to 1
  for (Eigen::Index c = 0; c < pred.cols(); ++c) {
    EXPECT_NEAR(pred.col(c).sum(), 1.0, 1e-9);
  }
}

TEST(NeuralNetworkNewAPI, TrainsOnToyXOR) {
  // XOR with 2 hidden ReLU, 1 output Sigmoid — should improve from random
  Eigen::MatrixXd X(2, 4);
  X << 0, 0, 1, 1,
       0, 1, 0, 1;
  Eigen::VectorXd Y(4);
  Y << 0, 1, 1, 0;

  NeuralNetwork net(std::vector<Layers::Layer>{
      Layers::Dense(2, 4, Activation::ReLU),
      Layers::Dense(4, 1, Activation::Sigmoid),
  });
  double acc_before = net.accuracy(X, Y);
  net.train(X, Y, 0.5, 100);
  double acc_after = net.accuracy(X, Y);
  EXPECT_GE(acc_after, acc_before);
  Eigen::MatrixXd pred = net.predict(X);
  EXPECT_TRUE(pred.allFinite());
}

TEST(NeuralNetworkLegacyShim, StillBuilds) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  NeuralNetwork net(std::vector<FlexNN::Layer>{
      FlexNN::Layer(2, 4, "relu"), FlexNN::Layer(4, 1, "sigmoid")});
#pragma GCC diagnostic pop
  Eigen::MatrixXd X(2, 4);
  X << 0, 0, 1, 1,
       0, 1, 0, 1;
  Eigen::VectorXd Y(4);
  Y << 0, 1, 1, 0;
  EXPECT_NO_THROW(net.train(X, Y, 0.5, 5));
  Eigen::MatrixXd pred = net.predict(X);
  EXPECT_TRUE(pred.allFinite());
  EXPECT_EQ(pred.rows(), 1);
  EXPECT_EQ(pred.cols(), 4);
}

TEST(NeuralNetworkLegacyShim, PreservesSoftmax) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  NeuralNetwork net(std::vector<FlexNN::Layer>{
      FlexNN::Layer(2, 4, "relu"), FlexNN::Layer(4, 2, "softmax")});
#pragma GCC diagnostic pop
  ASSERT_EQ(net.layers().size(), 2u);
  EXPECT_EQ(net.layers()[0].activation(), Activation::ReLU);
  EXPECT_EQ(net.layers()[1].activation(), Activation::Softmax);
  EXPECT_EQ(net.layers()[0].type(), Layers::LayerType::Dense);
}
