/**
 * @file FlexNN.h
 * @brief Header file for the FlexNN neural network library.
 *
 * This library provides a flexible neural network implementation using Eigen for matrix operations.
 * It includes classes for layers and the neural network itself, allowing for easy construction,
 * training, and prediction.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef FlexNN_H
#define FlexNN_H

#include <vector>
#include <Eigen/Dense>

#include "Layer.h"

/**
 * @namespace FlexNN
 * @brief Namespace for the FlexNN neural network library.
 *
 * This namespace contains all the classes and functions related to the FlexNN library,
 * including the NeuralNetwork class and Layer class. It provides a structured way to organize
 * the library's components and avoid naming conflicts with other libraries.
 */
namespace FlexNN
{
  /**
   * @class NeuralNetwork
   * @brief Class representing a neural network.
   *
   * This class encapsulates the functionality of a neural network, including training,
   * prediction, and accuracy calculation. It uses a vector of Layer objects to represent
   * the structure of the network.
   */
  class NeuralNetwork
  {
  public:
    /**
     * @brief Constructor for the NeuralNetwork class.
     *
     * @param layers A vector of Layer objects representing the layers of the neural network.
     */
    NeuralNetwork(const std::vector<Layer> &layers) : layers(layers) {}

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
    void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, double learningRate, int epochs);

    /**
     * @brief Calculate the accuracy of the neural network.
     *
     * This method computes the accuracy of the neural network's predictions against the target data.
     *
     * @param X The input data for prediction.
     * @param Y The target output data for comparison.
     * @return The accuracy as a double value.
     */
    double accuracy(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);

    /**
     * @brief Predict the output for given input data.
     *
     * This method performs a forward pass through the neural network to predict the output
     * for the provided input data.
     *
     * @param input The input data for prediction.
     * @return The predicted output as an Eigen::MatrixXd.
     */
    Eigen::MatrixXd predict(const Eigen::MatrixXd &input)
    {
      auto outputs = forward(input);
      return outputs.back(); // Return the final output (activation of the last layer)
    }

  private:
    /**
     * @brief A vector of Layer objects representing the layers of the neural network.
     *
     * This vector holds all the layers in the neural network, allowing for flexible
     * architecture and easy manipulation of the network structure.
     */
    std::vector<Layer> layers;

    /**
     * @brief Forward pass through the neural network.
     *
     * This method performs a forward pass through all layers of the neural network,
     * computing the activations for each layer based on the input data.
     *
     * @param input The input data for the forward pass.
     * @return A vector of Eigen::MatrixXd containing the outputs of each layer.
     */
    std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd &input);

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
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd> &outputs, const Eigen::MatrixXd &target);

    /**
     * @brief Update the weights of the neural network.
     *
     * This method updates the weights of each layer based on the calculated gradients
     * and the specified learning rate.
     *
     * @param gradients A vector of Eigen::MatrixXd containing the gradients for each layer.
     * @param learningRate The learning rate for updating weights.
     */
    void updateWeights(const std::vector<Eigen::MatrixXd> &gradients, double learningRate);
  };
}

#endif // FlexNN_H