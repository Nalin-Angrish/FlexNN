/**
 * @file pool_test.cpp
 * @brief Tests for MaxPool1D / AvgPool1D forward, propagate, and ModelIO.
 */

#include <gtest/gtest.h>

#include "layers/Pool1D.hpp"
#include "layers/Layer.hpp"
#include "FlexNN.h"
#include "ModelIO.hpp"
#include "activations/Activation.hpp"

using namespace FlexNN::Layers;
using namespace FlexNN::Activations;

TEST(MaxPoolForward, Basic) {
  Pool1DParams p{1, 2, 2, 0}; // C=1, K=2, stride2, pad0
  MaxPool1D pool(p, Activation::None);
  Eigen::MatrixXd X(4, 1); // L_in 4
  X << 1, 3, 2, 4;
  auto [Z, A] = pool.forward(X);
  // L_out = (4-2)/2+1=2, Z = [max(1,3)=3, max(2,4)=4]
  EXPECT_EQ(Z.rows(), 2);
  EXPECT_EQ(Z(0, 0), 3);
  EXPECT_EQ(Z(1, 0), 4);
}

TEST(AvgPoolForward, Basic) {
  Pool1DParams p{1, 2, 2, 0};
  AvgPool1D pool(p, Activation::None);
  Eigen::MatrixXd X(4, 1);
  X << 1, 3, 2, 4;
  auto [Z, A] = pool.forward(X);
  EXPECT_DOUBLE_EQ(Z(0, 0), 2.0); // (1+3)/2
  EXPECT_DOUBLE_EQ(Z(1, 0), 3.0); // (2+4)/2
}

TEST(PoolVariant, InLayer) {
  Pool1DParams p{2, 2, 2, 0};
  MaxPool1D m(p, Activation::None);
  AvgPool1D a(p, Activation::None);
  Layer lm(m);
  Layer la(a);
  EXPECT_EQ(lm.type(), LayerType::MaxPool1D);
  EXPECT_EQ(la.type(), LayerType::AvgPool1D);
  ASSERT_NE(lm.asMaxPool1D(), nullptr);
  ASSERT_NE(la.asAvgPool1D(), nullptr);
}

TEST(PoolPropagate, Max) {
  Pool1DParams p{1, 2, 2, 0};
  MaxPool1D pool(p, Activation::None);
  Eigen::MatrixXd X(4, 1);
  X << 1, 3, 2, 4;
  auto [Z, A] = pool.forward(X);
  Eigen::MatrixXd dZ(2, 1);
  dZ << 1, 1;
  Eigen::MatrixXd dX = pool.propagate(dZ);
  // MaxPool routes to argmax: positions 1 and 3 (values 3 and 4)
  EXPECT_EQ(dX.rows(), 4);
  EXPECT_DOUBLE_EQ(dX(0, 0), 0);
  EXPECT_DOUBLE_EQ(dX(1, 0), 1);
  EXPECT_DOUBLE_EQ(dX(2, 0), 0);
  EXPECT_DOUBLE_EQ(dX(3, 0), 1);
}

TEST(PoolPropagate, Avg) {
  Pool1DParams p{1, 2, 2, 0};
  AvgPool1D pool(p, Activation::None);
  Eigen::MatrixXd X(4, 1);
  X << 1, 3, 2, 4;
  auto [Z, A] = pool.forward(X);
  Eigen::MatrixXd dZ(2, 1);
  dZ << 1, 1;
  Eigen::MatrixXd dX = pool.propagate(dZ);
  // Avg distributes 0.5 each
  EXPECT_DOUBLE_EQ(dX(0, 0), 0.5);
  EXPECT_DOUBLE_EQ(dX(1, 0), 0.5);
  EXPECT_DOUBLE_EQ(dX(2, 0), 0.5);
  EXPECT_DOUBLE_EQ(dX(3, 0), 0.5);
}

TEST(PoolModelIO, RoundTrip) {
  FlexNN::NeuralNetwork net(std::vector<FlexNN::Layers::Layer>{
      MaxPool1D(Pool1DParams{2, 2, 2, 0}, Activation::None),
      AvgPool1D(Pool1DParams{2, 2, 2, 0}, Activation::None),
  });
  const std::string path = "/tmp/flexnn_pool_roundtrip.bin";
  auto st = FlexNN::exportModel(net, path);
  ASSERT_TRUE(st.ok) << st.error;
  FlexNN::NeuralNetwork net2(std::vector<FlexNN::Layers::Layer>{Dense(1, 1)});
  auto st2 = FlexNN::importModel(net2, path);
  ASSERT_TRUE(st2.ok) << st2.error;
  ASSERT_EQ(net2.layers().size(), 2u);
  EXPECT_EQ(net2.layers()[0].type(), LayerType::MaxPool1D);
  EXPECT_EQ(net2.layers()[1].type(), LayerType::AvgPool1D);
  std::remove(path.c_str());
}

TEST(PoolNetwork, ConvPoolDense) {
  Eigen::MatrixXd X(2 * 8, 2);
  X.setRandom();
  Eigen::VectorXd Y(2);
  Y << 0, 1;
  FlexNN::NeuralNetwork net(std::vector<FlexNN::Layers::Layer>{
      Conv1D(Conv1DParams{2, 2, 3, 1, 1, 1}, Activation::ReLU), // 2*8 ->2*8
      MaxPool1D(Pool1DParams{2, 2, 2, 0}, Activation::None),    // 2*8 ->2*4
      Dense(2 * 4, 2, Activation::Softmax),
  });
  EXPECT_NO_THROW(net.train(X, Y, 0.1, 2));
  Eigen::MatrixXd pred = net.predict(X);
  EXPECT_TRUE(pred.allFinite());
}
