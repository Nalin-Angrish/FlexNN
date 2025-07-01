/**
 * @file Utility.h
 * @brief Utility functions for FlexNN neural network library.
 *
 * This file contains utility functions for one-hot encoding, reading CSV files,
 * and splitting datasets into training, validation, and test sets.
 *
 * Will add more utility functions as needed.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef FlexNN_UTILITY_H
#define FlexNN_UTILITY_H

#include <Eigen/Dense>
#include <string>
#include <vector>

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
   * @brief One-hot encodes a vector of class labels.
   *
   * This function takes a vector of class labels and converts it into a one-hot encoded matrix.
   * Each row corresponds to a class label, and each column corresponds to a class.
   *
   * @param Y The input vector of class labels.
   * @param num_classes The number of unique classes.
   * @return An Eigen::MatrixXd where each row is a one-hot encoded vector for the corresponding class label.
   */
  Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd &Y, int num_classes);

  /**
   * @brief Reads a CSV file and splits it into features (X) and labels (Y).
   *
   * This function reads a CSV file where the first column is considered the label (Y)
   * and the remaining columns are considered features (X). It populates the provided
   * Eigen matrices with the data from the CSV file.
   *
   * @param filename The path to the CSV file to read.
   * @param X The Eigen::MatrixXd to store the features (all columns except the first).
   * @param Y The Eigen::VectorXd to store the labels (the first column).
   */
  void readCSV_XY(const std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y);

  /**
   * @brief Splits the dataset into multiple sets based on specified proportions.
   *
   * This function takes a dataset represented by features (X) and labels (Y),
   * and splits it into multiple sets according to the provided proportions.
   *
   * @param X The input feature matrix (Eigen::MatrixXd).
   * @param Y The input label vector (Eigen::VectorXd).
   * @param proportions A vector of doubles representing the proportions for each split.
   * @return A vector of pairs, where each pair contains a feature matrix and a label vector for each split.
   */
  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>
  splitXY(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, const std::vector<double> &proportions);
}

#endif // FlexNN_UTILITY_H