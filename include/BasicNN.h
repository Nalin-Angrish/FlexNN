#ifndef BasicNN_H
#define BasicNN_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "Layer.h"
#include "Utility.h"

namespace BasicNN
{
  class NeuralNetwork
  {
  public:
    // Constructor
    NeuralNetwork(const std::vector<Layer> &layers) : layers(layers) {}

    void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target, double learningRate, int epochs)
    {
      Eigen::MatrixXd Y_onehot = BasicNN::oneHotEncode(target, target.maxCoeff() + 1); // Convert target to one-hot encoding
      for (int epoch = 0; epoch < epochs; ++epoch)
      {
        auto outputs = forward(input);
        auto gradients = backward(outputs, Y_onehot);
        updateWeights(gradients, learningRate);
        if((epoch+1) % 10 == 0) {
          std::cout << "Epoch " << epoch+1 << "/" << epochs <<": Accuracy = " << this->accuracy(input, target) << std::endl;
        }
      }
    }

    Eigen::MatrixXd predict(const Eigen::MatrixXd &input)
    {
      auto outputs = forward(input);
      return outputs.back(); // Return the final output (activation of the last layer)
    }

    double accuracy(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y){
      Eigen::MatrixXd predictions = this->predict(X);
      int correct = 0;
      for (int i = 0; i < predictions.cols(); ++i) {
          int predictedClass;
          predictions.col(i).maxCoeff(&predictedClass);
          if (predictedClass == static_cast<int>(Y(i))) {
              correct++;
          }
      }
      return static_cast<double>(correct) / predictions.cols();
    }

  private:
    std::vector<Layer> layers; // Layers of the neural network

    // Forward pass method
    std::vector<Eigen::MatrixXd> forward(const Eigen::MatrixXd &input)
    {
      std::vector<Eigen::MatrixXd> outputs;
      outputs.push_back(input); // Start with the input as the first output
      for (size_t i = 0; i < layers.size(); ++i)
      {
        auto result = layers[i].forward(outputs[outputs.size() - 1]);
        outputs.push_back(result.first);
        outputs.push_back(result.second); // Store both Z and A
      }
      return outputs;
    }

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd> &outputs, const Eigen::MatrixXd &target)
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

    void updateWeights(const std::vector<Eigen::MatrixXd> &gradients, double learningRate)
    {
      for (int i = 0; i < layers.size(); ++i)
      {
        Eigen::MatrixXd dW = gradients[2 * i];
        Eigen::VectorXd db = gradients[2 * i + 1];
        layers[i].updateWeights(dW, db, learningRate);
      }
    }
  };
}

#endif // BasicNN_H