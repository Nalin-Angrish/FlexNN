/**
 * @file Layer.h
 * @brief Header file for the Layer class in the FlexNN neural network library.
 *
 * This file defines the Layer class, which represents a single layer in a neural network.
 * It includes methods for forward and backward passes, weight updates, and accessing weights and biases.
 * The Layer class uses Eigen for matrix operations and supports various activation functions.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef FlexNN_Layer_H
#define FlexNN_Layer_H

#include <utility>
#include <Eigen/Dense>

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
   * @class Layer
   * @brief Represents a single layer in a neural network.
   *
   * The Layer class encapsulates the properties and methods required for a neural network layer,
   * including weights, biases, forward and backward passes, and weight updates.
   *
   * This class is designed to be flexible and can be used with different activation functions.
   * It supports both relu and softmax activation functions by default, but can be extended to include others.
   */
  class Layer
  {
  public:
    /**
     * @brief Constructor for the Layer class.
     *
     * Initializes the layer with random weights and biases.
     *
     * @param inputSize The size of the input to this layer.
     * @param outputSize The size of the output from this layer (also the number of neurons of this layer).
     * @param activationFunction The activation function to be used in this layer (default is "relu").
     *
     * @note If this is the last layer, the activation function should be "softmax".
     */
    Layer(int inputSize, int outputSize, const std::string &activationFunction = "relu")
        : inputSize(inputSize), outputSize(outputSize), activationFunction(activationFunction)
    {
      // Initialize weights and biases
      W = Eigen::MatrixXd::Random(outputSize, inputSize) * 0.5;
      b = Eigen::VectorXd::Random(outputSize) * 0.5;
    }

    /**
     * @brief Getters for weights.
     *
     * These methods return the weights of the layer.
     *
     * @return Eigen::MatrixXd The weights of the layer.
     */
    Eigen::MatrixXd getWeights() const
    {
      return W; // Return the weights of the layer
    }

    /**
     * @brief Getters for biases.
     *
     * These methods return the biases of the layer.
     *
     * @return Eigen::VectorXd The biases of the layer.
     */
    Eigen::VectorXd getBiases() const
    {
      return b; // Return the biases of the layer
    }

    /**
     * @brief Update weights and biases.
     *
     * This method updates the weights and biases of the layer using the provided gradients
     * and a specified learning rate.
     *
     * @param dW The gradient of the weights.
     * @param db The gradient of the biases.
     * @param learningRate The learning rate for updating the weights and biases.
     */
    void updateWeights(const Eigen::MatrixXd &dW, const Eigen::VectorXd &db, double learningRate)
    {
      W -= learningRate * dW; // Update weights
      b -= learningRate * db; // Update biases
    }

    /**
     * @brief Forward pass through the layer.
     *
     * This method computes the output of the layer given an input matrix.
     * It applies the activation function to the linear combination of inputs and weights.
     *
     * @param input The input data for the forward pass.
     * @return A pair containing the linear output (Z) and the activated output (A).
     */
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(const Eigen::MatrixXd &input);

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
    Eigen::MatrixXd backward(const Eigen::MatrixXd &nextW, const Eigen::MatrixXd &nextdZ, const Eigen::MatrixXd &currZ);

  private:
    /**
     * @brief Input layer size.
     */
    int inputSize;
    /**
     * @brief Output layer size (number of neurons in this layer).
     */
    int outputSize;
    /**
     * @brief Activation function used in this layer.
     *
     * This can be "relu" or "softmax" for now, will implement more later.
     */
    std::string activationFunction;
    /**
     * @brief Weights of the layer.
     *
     * This is a matrix where each row corresponds to a neuron in this layer
     * and each column corresponds to an input feature.
     */
    Eigen::MatrixXd W;
    /**
     * @brief Biases of the layer.
     *
     * This is a vector where each element corresponds to a neuron in this layer.
     */
    Eigen::VectorXd b;
  };
}

#endif // FlexNN_Layer_H