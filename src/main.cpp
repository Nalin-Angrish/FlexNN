/**
 * @file main.cpp
 * @brief Main file for the MNIST digit recognition example using FlexNN.
 *
 * This file demonstrates how to use the FlexNN library to create, train, and evaluate a neural network
 * for recognizing handwritten digits from the MNIST dataset. It includes reading the dataset from a CSV file,
 * normalizing the data, splitting it into training and test sets, defining the neural network architecture,
 * training the network, and evaluating its performance. The user can also test the model with specific indices
 * from the test set to see the predicted and actual labels, along with an ASCII representation of the image.
 */
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "FlexNN.h"
#include "Utility.h"

/**
 * @brief Main function to demonstrate a simple neural network for MNIST digit recognition.
 *
 * This program reads the MNIST dataset from a CSV file, normalizes the data, splits it into training and test sets,
 * creates a neural network with two layers, trains the network on the training data, and evaluates its accuracy on both training and test sets.
 * It also allows the user to input an index to test the model's prediction on a specific sample from the test set.
 * The predicted label and actual label are displayed, along with an ASCII representation of the image.
 */
int main()
{
  Eigen::MatrixXd X;
  Eigen::VectorXd Y;
  // Read the MNIST dataset from a CSV file
  std::cout << "Reading CSV file..." << std::endl;
  FlexNN::readCSV_XY("data/mnist-digit-recognition.csv", X, Y);
  X = X.array() / 255.0; // Normalize the input data

  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> data = FlexNN::splitXY(X, Y, {0.9, 0.1}); // Split data into training and test sets
  X = data[0].first;                                                                                 // Training set features
  Y = data[0].second;                                                                                // Training set labels
  Eigen::MatrixXd X_test = data[1].first;                                                            // Test set features
  Eigen::VectorXd Y_test = data[1].second;                                                           // Test set labels

  std::cout << "Data loaded successfully." << std::endl;
  std::cout << "Training data size: " << X.rows() << " samples, " << X.cols() << " features." << std::endl;
  std::cout << "Test data size: " << X_test.rows() << " samples, " << X_test.cols() << " features." << std::endl;

  // Transpose the matrices to match the expected input format for FlexNN
  // FlexNN expects input in the form (features, samples), so we transpose the matrices
  X.transposeInPlace();      // Now X is (features, samples)
  X_test.transposeInPlace(); // Now X_test is (features, samples)

  // Define the neural network architecture
  // Here we create a simple neural network with one hidden layer of 64 neurons and an output layer of 10 neurons (for digit classification)
  FlexNN::NeuralNetwork nn({FlexNN::Layer(X.rows(), 64, "relu"),
                            FlexNN::Layer(64, 10, "softmax")});
  std::cout << "Neural Network created with 2 layers." << std::endl;

  // Train the neural network
  // We use a learning rate of 0.5 and train for 300 epochs
  std::cout << "Training started." << std::endl;
  nn.train(X, Y, 0.5, 300);
  std::cout << "Training completed." << std::endl;

  // Evaluate the accuracy of the neural network on both training and test sets
  std::cout << "Accuracy on training data: " << nn.accuracy(X, Y) * 100 << "%" << std::endl;
  std::cout << "Accuracy on testing data: " << nn.accuracy(X_test, Y_test) * 100 << "%" << std::endl;

  // Allow the user to test the model with specific indices from the test set
  int testIndex;
  std::cout << ">> ";
  std::cin >> testIndex; // Wait for user input to proceed with testing
  while (testIndex)
  {
    if (testIndex < 0 || testIndex >= X_test.cols())
    {
      std::cout << "Invalid index. Please enter a number between 0 and " << X_test.cols() - 1 << "." << std::endl;
      std::cout << ">> ";
      std::cin >> testIndex; // Wait for user input to proceed with testing
      continue;
    }
    // Predict the label for the given test index
    Eigen::VectorXd prediction = nn.predict(X_test.col(testIndex));
    int predictedClass;
    prediction.maxCoeff(&predictedClass); // Get the index of the maximum value in the prediction vector
    std::cout << "Predicted Label: " << predictedClass << std::endl;
    std::cout << "Actual Label: " << Y_test(testIndex) << std::endl;

    // Display the image as ASCII art
    const Eigen::VectorXd &img = X_test.col(testIndex);
    std::cout << "Image:" << std::endl;
    for (int i = 0; i < 28; ++i)
    {
      for (int j = 0; j < 28; ++j)
      {
        double pixel = img(i * 28 + j) * 255.0; // If normalized, multiply back
        char c;
        if (pixel > 200)
          c = '#';
        else if (pixel > 120)
          c = '*';
        else if (pixel > 50)
          c = '.';
        else
          c = ' ';
        std::cout << c;
      }
      std::cout << std::endl;
    }
    std::cout << ">> ";
    std::cin >> testIndex; // Wait for user input to proceed with the next test
  }
  return 0;
}