#include <utility>
#include <Eigen/Dense>

#include "Layer.h"

// Forward pass method
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> BasicNN::Layer::forward(const Eigen::MatrixXd &input)
{
  Eigen::MatrixXd output = (W * input).colwise() + b; // Linear transformation
  Eigen::MatrixXd activation;
  if (activationFunction == "relu")
  {
    activation = output.unaryExpr([](double x)
                                  { return std::max(0.0, x); }); // ReLU activation
  }
  else if (activationFunction == "softmax")
  {
    // Numerically stable softmax, applied column-wise
    activation = output;
    for (int i = 0; i < output.cols(); ++i)
    {
      Eigen::VectorXd col = output.col(i);
      double maxCoeff = col.maxCoeff();
      Eigen::VectorXd expCol = (col.array() - maxCoeff).exp();
      activation.col(i) = expCol / expCol.sum();
    }
  }
  else
  {
    activation = output; // No activation function, just return the linear output
  }
  return std::make_pair(output, activation); // Z, A
}

// Backward pass method, returns dZ
Eigen::MatrixXd BasicNN::Layer::backward(const Eigen::MatrixXd &nextW, const Eigen::MatrixXd &nextdZ, const Eigen::MatrixXd &currZ)
{
  Eigen::MatrixXd dZ;
  if (activationFunction == "relu")
  {
    // Derivative of ReLU: 1 if currZ > 0, else 0
    dZ = (nextW.transpose() * nextdZ).array() * (currZ.array() > 0.0).cast<double>();
  }
  else if (activationFunction == "softmax")
  {
    Eigen::VectorXd expZ = currZ.array().exp();
    dZ = (nextW.transpose() * nextdZ) * (expZ / expZ.sum()).matrix(); // Gradient for Softmax
  }
  else
  {
    dZ = nextW.transpose() * nextdZ; // No activation function, just pass the gradient
  }
  return dZ; // Return the gradient of Z
}