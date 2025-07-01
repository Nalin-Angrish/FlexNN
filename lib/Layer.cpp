/**
 * @file Layer.cpp
 * @brief Source file for the Layer class in the FlexNN neural network library.
 *
 * This file defines the Layer class, which represents a single layer in a neural network.
 * It includes methods for forward and backward passes, weight updates, and accessing weights and biases.
 * The Layer class uses Eigen for matrix operations and supports various activation functions.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include <utility>
#include <Eigen/Dense>

#include "Layer.h"

/**
 * @brief Forward pass through the layer.
 *
 * This method computes the output of the layer given an input matrix.
 * It applies the activation function to the linear combination of inputs and weights.
 *
 * @param input The input data for the forward pass.
 * @return A pair containing the linear output (Z) and the activated output (A).
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> FlexNN::Layer::forward(const Eigen::MatrixXd &input)
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

/**
 * @brief Backward pass through the layer.
 *
 * This method computes the gradient of the loss with respect to the inputs of this layer
 * given the gradients from the next layer.
 *
 * @param nextW The weights of the next layer.
 * @param nextdZ The gradients from the next layer.
 * @param currZ The linear output (Z) of this layer.
 * @return The gradient of the loss with respect to the inputs of this layer (dZ).
 */
Eigen::MatrixXd FlexNN::Layer::backward(const Eigen::MatrixXd &nextW, const Eigen::MatrixXd &nextdZ, const Eigen::MatrixXd &currZ)
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