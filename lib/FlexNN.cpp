/**
 * @file FlexNN.cpp
 * @brief Source file for the FlexNN neural network library.
 *
 * This library provides a flexible neural network implementation using Eigen for matrix operations.
 * It includes classes for layers and the neural network itself, allowing for easy construction,
 * training, and prediction.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "FlexNN.h"
#include "Utility.h"

/**
 * @brief Train the neural network.
 *
 * This method trains the neural network using the provided input and target data.
 * It performs forward and backward passes, updating weights based on the gradients.
 *
 * @param input The input data for training.
 * @param target The target output data for training.
 * @param learningRate The learning rate for weight updates.
 * @param epochs The number of training epochs.
 */
void FlexNN::NeuralNetwork::train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, double learningRate, int epochs)
{
  Eigen::MatrixXd Y_onehot = FlexNN::oneHotEncode(target, target.maxCoeff() + 1); // Convert target to one-hot encoding
  for (int epoch = 0; epoch < epochs; ++epoch)                                    // for each epoch
  {
    auto outputs = forward(input);                // Perform forward pass to compute outputs
    auto gradients = backward(outputs, Y_onehot); // Perform backward pass to compute gradients
    updateWeights(gradients, learningRate);       // Update weights based on gradients
    if ((epoch + 1) % 10 == 0)                    // Log the accuracy every 10 epochs for debugging
    {
      std::cout << "Epoch " << epoch + 1 << "/" << epochs << ": Accuracy = " << this->accuracy(input, target) << std::endl;
    }
  }
}

/**
 * @brief Calculate the accuracy of the neural network.
 *
 * This method computes the accuracy of the neural network's predictions against the target data.
 *
 * @param X The input data for prediction.
 * @param Y The target output data for comparison.
 * @return The accuracy as a double value.
 */
double FlexNN::NeuralNetwork::accuracy(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
{
  Eigen::MatrixXd predictions = this->predict(X); // Get predictions from the neural network
  int correct = 0;
  for (int i = 0; i < predictions.cols(); ++i) // Iterate through each prediction
  {
    int predictedClass;
    predictions.col(i).maxCoeff(&predictedClass);
    if (predictedClass == static_cast<int>(Y(i)))
    {
      correct++; // Increment correct count if prediction matches the target class
    }
  }
  return static_cast<double>(correct) / predictions.cols(); // Calculate accuracy as the ratio of correct predictions to total predictions
}

/**
 * @brief Forward pass through the neural network.
 *
 * This method performs a forward pass through all layers of the neural network,
 * computing the activations for each layer based on the input data.
 *
 * @param input The input data for the forward pass.
 * @return A vector of Eigen::MatrixXd containing the outputs of each layer.
 */
std::vector<Eigen::MatrixXd> FlexNN::NeuralNetwork::forward(const Eigen::MatrixXd &input)
{
  std::vector<Eigen::MatrixXd> outputs;
  outputs.push_back(input); // Start with the input as the first output
  for (size_t i = 0; i < layers.size(); ++i)
  {
    auto result = layers[i].forward(outputs[outputs.size() - 1]); // Forward pass through the layer
    outputs.push_back(result.first);
    outputs.push_back(result.second); // Store both Z and A
  }
  return outputs; // Return all outputs including activations and pre-activations
}

/**
 * @brief Backward pass through the neural network.
 *
 * This method performs a backward pass through the neural network, calculating
 * the gradients for each layer based on the outputs and target data.
 *
 * @param outputs The outputs from the forward pass.
 * @param target The target output data for training.
 * @return A vector of Eigen::MatrixXd containing the gradients for each layer.
 */
std::vector<Eigen::MatrixXd> FlexNN::NeuralNetwork::backward(const std::vector<Eigen::MatrixXd> &outputs, const Eigen::MatrixXd &target)
{
  std::vector<Eigen::MatrixXd> gradients;
  std::vector<Eigen::MatrixXd> dZs; // To store dZ for each layer

  Eigen::MatrixXd dZ = outputs.back() - target;                          // Compute the initial dZ (gradient of the loss w.r.t. output)
  dZs.push_back(dZ);                                                     // Store dZ for this layer
  int m = dZ.cols();                                                     // Number of examples
  gradients.push_back(dZ.rowwise().mean());                              // Store both dW and db
  gradients.push_back(dZ * outputs[outputs.size() - 3].transpose() / m); // dW

  for (int i = layers.size() - 2; i >= 0; --i)
  {
    dZ = layers[i].backward(layers[i + 1].getWeights(), dZs.back(), outputs[2 * i + 1]);
    dZs.push_back(dZ);                                        // Store dZ for this layer
    gradients.push_back(dZ.rowwise().mean());                 // Store both db
    gradients.push_back(dZ * outputs[2 * i].transpose() / m); // dW
  }

  std::reverse(gradients.begin(), gradients.end()); // Reverse the order of gradients to match layer order
  // gradients now contains dW and db for each layer in the correct order
  return gradients;
}

/**
 * @brief Update the weights of the neural network.
 *
 * This method updates the weights of each layer based on the calculated gradients
 * and the specified learning rate.
 *
 * @param gradients A vector of Eigen::MatrixXd containing the gradients for each layer.
 * @param learningRate The learning rate for updating weights.
 */
void FlexNN::NeuralNetwork::updateWeights(const std::vector<Eigen::MatrixXd> &gradients, double learningRate)
{
  for (int i = 0; i < layers.size(); ++i)
  {
    Eigen::MatrixXd dW = gradients[2 * i];
    Eigen::VectorXd db = gradients[2 * i + 1];
    layers[i].updateWeights(dW, db, learningRate); // Update weights and biases of the layer
  }
}