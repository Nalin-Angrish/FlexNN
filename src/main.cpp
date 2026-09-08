/**
 * @file main.cpp
 * @brief MNIST example — now uses FlexNN::Layers::Dense + Activations + deterministic seed.
 *
 * Demonstrates the new modular API: `FlexNN::Layers::Dense` with
 * `FlexNN::Activations::Activation`, `setRandomSeed`, and `splitXY` with
 * explicit seed. The old stringly-typed `FlexNN::Layer(..., "relu")` is
 * deprecated but still builds via the shim until removed.
 */

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "FlexNN.h"
#include "Utility.h"
#include "activations/Activation.hpp"
#include "layers/Dense.hpp"

int main() {
  Eigen::MatrixXd X;
  Eigen::VectorXd Y;
  std::cout << "Reading CSV file..." << std::endl;
  FlexNN::readCSV_XY("data/mnist-digit-recognition.csv", X, Y);
  X = X.array() / 255.0; // Normalize to [0,1]

  // Deterministic split — same train/test every run with seed 42.
  // Uses the new overload `splitXY(X,Y,props,seed)` which is hermetic.
  FlexNN::setRandomSeed(42);
  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> data =
      FlexNN::splitXY(X, Y, {0.9, 0.1}, 42);
  X = data[0].first;
  Y = data[0].second;
  Eigen::MatrixXd X_test = data[1].first;
  Eigen::VectorXd Y_test = data[1].second;

  std::cout << "Data loaded successfully." << std::endl;
  std::cout << "Training data size: " << X.rows() << " samples, " << X.cols() << " features." << std::endl;
  std::cout << "Test data size: " << X_test.rows() << " samples, " << X_test.cols() << " features." << std::endl;

  // FlexNN expects [features × samples], so transpose after split which is [samples × features]
  X.transposeInPlace();
  X_test.transposeInPlace();

  // New API: Layers::Dense with enum activation (no heap, -Wswitch-enum checked).
  // The old `FlexNN::Layer(X.rows(),64,"relu")` still works but is deprecated.
  using namespace FlexNN;
  NeuralNetwork nn({
      Layers::Dense(X.rows(), 64, Activations::Activation::ReLU),
      Layers::Dense(64, 10, Activations::Activation::Softmax),
  });
  std::cout << "Neural Network created with 2 layers (Dense 64 ReLU + Dense 10 Softmax)." << std::endl;

  std::cout << "Training started." << std::endl;
  nn.train(X, Y, 0.5, 300);
  std::cout << "Training completed." << std::endl;

  std::cout << "Accuracy on training data: " << nn.accuracy(X, Y) * 100 << "%" << std::endl;
  std::cout << "Accuracy on testing data: " << nn.accuracy(X_test, Y_test) * 100 << "%" << std::endl;

  // Optional: save model via ModelIO (new in PR-06)
  // auto st = exportModel(nn, "model.bin");
  // if (!st.ok) std::cerr << "export failed: " << st.error << "\n";

  int testIndex;
  std::cout << ">> ";
  std::cin >> testIndex;
  while (testIndex) {
    if (testIndex < 0 || testIndex >= X_test.cols()) {
      std::cout << "Invalid index. Please enter a number between 0 and " << X_test.cols() - 1 << "." << std::endl;
      std::cout << ">> ";
      std::cin >> testIndex;
      continue;
    }
    Eigen::VectorXd prediction = nn.predict(X_test.col(testIndex));
    int predictedClass;
    prediction.maxCoeff(&predictedClass);
    std::cout << "Predicted Label: " << predictedClass << std::endl;
    std::cout << "Actual Label: " << Y_test(testIndex) << std::endl;

    const Eigen::VectorXd &img = X_test.col(testIndex);
    std::cout << "Image:" << std::endl;
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        double pixel = img(i * 28 + j) * 255.0;
        char c;
        if (pixel > 200) c = '#';
        else if (pixel > 120) c = '*';
        else if (pixel > 50) c = '.';
        else c = ' ';
        std::cout << c;
      }
      std::cout << std::endl;
    }
    std::cout << ">> ";
    std::cin >> testIndex;
  }
  return 0;
}
