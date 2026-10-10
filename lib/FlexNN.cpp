/**
 * @file FlexNN.cpp
 * @brief Implements FlexNN::NeuralNetwork with heterogeneous Layers::Layer variant.
 *
 * After PR-04, the network holds `vector<Layers::Layer>` where each `Layer`
 * is a `variant<Dense, ...>` (currently only Dense). Forward/backward use
 * `std::visit` via `Layer::forward/backward`. The last-layer Softmax +
 * cross-entropy is fused as `dZ = (A - Y)/m` in `backward()` — never calling
 * `Dense::backward` for the last layer when it is Softmax, which fixes the
 * legacy `lib/Layer.cpp:72` Jacobian bug.
 *
 * The legacy `NeuralNetwork(vector<Layer>)` (stringly-typed Dense) is kept
 * as a deprecated shim that converts each old `Layer` to `Layers::Dense`
 * via `Activations::try_parse`.
 */

#include "FlexNN.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "Utility.h"
#include "activations/Activation.hpp"

namespace FlexNN {

// Legacy shim — converts old stringly-typed Dense layers to new enum-typed Dense.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
NeuralNetwork::NeuralNetwork(const std::vector<Layer>& oldLayers) {
  layers_.reserve(oldLayers.size());
  for (const auto& old : oldLayers) {
    int in = static_cast<int>(old.getWeights().cols());
    int out = static_cast<int>(old.getWeights().rows());
    // Preserve the requested activation string via the new getter
    Activations::Activation act = Activations::Activation::ReLU;
    std::string s = old.getActivationFunction();
    Activations::Activation parsed;
    if (Activations::try_parse(s, parsed)) {
      act = parsed;
    } else {
      act = Activations::Activation::None;
    }
    Layers::Dense d(in, out, act);
    d.setWeights(old.getWeights());
    d.setBiases(old.getBiases());
    layers_.emplace_back(std::move(d));
  }
}
#pragma GCC diagnostic pop

void NeuralNetwork::train(const Eigen::MatrixXd& input,
                          const Eigen::VectorXd& target, double learningRate,
                          int epochs) {
  // One-hot encode target once. `target` is label vector (size = batch)
  Eigen::MatrixXd Y_onehot = FlexNN::oneHotEncode(target, static_cast<int>(target.maxCoeff()) + 1);
  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto outputs = forward(input);
    auto gradients = backward(outputs, Y_onehot);
    updateWeights(gradients, learningRate);
    if ((epoch + 1) % 10 == 0) {
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << ": Accuracy = " << this->accuracy(input, target) << std::endl;
    }
  }
}

double NeuralNetwork::accuracy(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) {
  Eigen::MatrixXd predictions = this->predict(X);
  int correct = 0;
  for (Eigen::Index i = 0; i < predictions.cols(); ++i) {
    int predictedClass;
    predictions.col(i).maxCoeff(&predictedClass);
    if (predictedClass == static_cast<int>(Y(i))) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / predictions.cols();
}

std::vector<Eigen::MatrixXd> NeuralNetwork::forward(const Eigen::MatrixXd& input) {
  std::vector<Eigen::MatrixXd> outputs;
  outputs.reserve(1 + 2 * layers_.size());
  outputs.push_back(input);
  for (size_t i = 0; i < layers_.size(); ++i) {
    // Dispatch via Layers::Layer::forward (std::visit)
    auto result = layers_[i].forward(outputs.back());
    outputs.push_back(result.first);  // Z
    outputs.push_back(result.second); // A
  }
  return outputs;
}

std::vector<Eigen::MatrixXd> NeuralNetwork::backward(
    const std::vector<Eigen::MatrixXd>& outputs, const Eigen::MatrixXd& target) {
  std::vector<Eigen::MatrixXd> gradients;
  gradients.reserve(2 * layers_.size());
  std::vector<Eigen::MatrixXd> dZs;
  dZs.reserve(layers_.size());

  // Last-layer handling: if last activation is Softmax, fuse with CE as dZ = A - Y
  // (divided by batch later in dW). This matches the fix for lib/Layer.cpp:72.
  assert(!layers_.empty() && "NeuralNetwork requires at least one layer");
  Eigen::MatrixXd dZ;
  const auto lastAct = layers_.back().activation();
  if (lastAct == Activations::Activation::Softmax) {
    dZ = outputs.back() - target; // (A_last - Y_onehot)
  } else {
    // For non-Softmax last layer, treat as linear CE or use detail::backward
    // Here dZ = (A - Y) * f'(Z) — but since last is typically Softmax, this
    // branch is rare in v0.1. We compute dA = (A - Y) and then backward.
    Eigen::MatrixXd A_last = outputs.back();
    Eigen::MatrixXd Z_last = outputs[outputs.size() - 2];
    Eigen::MatrixXd dA = A_last - target;
    dZ = Activations::detail::backward(lastAct, dA, Z_last, A_last);
  }
  dZs.push_back(dZ);
  Eigen::Index m = dZ.cols();
  gradients.push_back(dZ.rowwise().mean()); // db_last
  gradients.push_back(dZ * outputs[outputs.size() - 3].transpose() / static_cast<double>(m)); // dW_last

  // Hidden layers reverse
  for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
    // Extract next layer's weight matrix for upstream dA = W_next^T * dZ_next
    Eigen::MatrixXd nextW = layers_[static_cast<size_t>(i) + 1].weightsMatrix();
    // If next layer is weightless (Pool/BN), weightsMatrix() is empty; then
    // the correct upstream is just dZ_next (pool/BN backward will have already
    // handled routing). In PR-04 only Dense exists, so nextW is always valid.
    if (nextW.size() == 0) {
      // Weightless next layer — upstream is just next dZ (already correctly shaped)
      dZ = layers_[static_cast<size_t>(i)].backward(
          Eigen::MatrixXd(), dZs.back(), outputs[2 * static_cast<size_t>(i) + 1]);
    } else {
      dZ = layers_[static_cast<size_t>(i)].backward(nextW, dZs.back(),
                                                     outputs[2 * static_cast<size_t>(i) + 1]);
    }
    dZs.push_back(dZ);
    gradients.push_back(dZ.rowwise().mean()); // db
    gradients.push_back(dZ * outputs[2 * static_cast<size_t>(i)].transpose() /
                        static_cast<double>(m)); // dW
  }

  std::reverse(gradients.begin(), gradients.end());
  return gradients;
}

void NeuralNetwork::updateWeights(const std::vector<Eigen::MatrixXd>& gradients,
                                  double learningRate) {
  for (size_t i = 0; i < layers_.size(); ++i) {
    Eigen::MatrixXd dW = gradients[2 * i];
    Eigen::VectorXd db = gradients[2 * i + 1];
    layers_[i].updateWeights(dW, db, learningRate);
  }
}

} // namespace FlexNN
