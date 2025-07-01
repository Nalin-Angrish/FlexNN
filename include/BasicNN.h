#ifndef BasicNN_H
#define BasicNN_H

#include <vector>
#include <Eigen/Dense>

#include "Layer.h"

namespace BasicNN
{
  class NeuralNetwork
  {
  public:
    // Constructor
    NeuralNetwork(const std::vector<Layer> &layers) : layers(layers) {}

    void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, double learningRate, int epochs);

    double accuracy(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);

    Eigen::MatrixXd predict(const Eigen::MatrixXd &input)
    {
      auto outputs = forward(input);
      return outputs.back(); // Return the final output (activation of the last layer)
    }
  private:
    std::vector<Layer> layers; // Layers of the neural network

    // Forward pass method
    std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd &input);

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd> &outputs, const Eigen::MatrixXd &target);

    void updateWeights(const std::vector<Eigen::MatrixXd> &gradients, double learningRate);
  };
}

#endif // BasicNN_H