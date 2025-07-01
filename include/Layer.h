#ifndef BasicNN_Layer_H
#define BasicNN_Layer_H

#include <utility>
#include <Eigen/Dense>

namespace BasicNN
{
  class Layer
  {
  public:
    // Constructor
    Layer(int inputSize, int outputSize, const std::string &activationFunction = "relu")
        : inputSize(inputSize), outputSize(outputSize), activationFunction(activationFunction)
    {
      // Initialize weights and biases
      W = Eigen::MatrixXd::Random(outputSize, inputSize) * 0.5;
      b = Eigen::VectorXd::Random(outputSize) * 0.5;
    }

    Eigen::MatrixXd getWeights() const
    {
      return W; // Return the weights of the layer
    }

    Eigen::VectorXd getBiases() const
    {
      return b; // Return the biases of the layer
    }

    void updateWeights(const Eigen::MatrixXd &dW, const Eigen::VectorXd &db, double learningRate)
    {
      W -= learningRate * dW; // Update weights
      b -= learningRate * db; // Update biases
    }

    // Forward pass method
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(const Eigen::MatrixXd &input);

    // Backward pass method, returns dZ
    Eigen::MatrixXd backward(const Eigen::MatrixXd &nextW, const Eigen::MatrixXd &nextdZ, const Eigen::MatrixXd &currZ);

  private:
    int inputSize;  // Size of the input to this layer
    int outputSize; // Size of the output from this layer
    std::string activationFunction;

    Eigen::MatrixXd W; // Weights of the layer
    Eigen::VectorXd b; // Biases of the layer
  };
}

#endif // BasicNN_Layer_H