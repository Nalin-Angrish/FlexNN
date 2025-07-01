/**
 * @file Utility.cpp
 * @brief Utility functions for FlexNN neural network library.
 *
 * This file contains utility functions for one-hot encoding, reading CSV files,
 * and splitting datasets into training, validation, and test sets.
 *
 * Will add more utility functions as needed.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

#include "Utility.h"

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
Eigen::MatrixXd FlexNN::oneHotEncode(const Eigen::VectorXd &Y, int num_classes)
{
  Eigen::MatrixXd Y_onehot = Eigen::MatrixXd::Zero(num_classes, Y.size()); // Initialize a zero matrix with num_classes rows and Y.size() columns
  for (int i = 0; i < Y.size(); ++i)
  {
    int label = static_cast<int>(Y(i));
    if (label >= 0 && label < num_classes)
      Y_onehot(label, i) = 1.0; // Set the corresponding position to 1
  }
  return Y_onehot;
}

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
void FlexNN::readCSV_XY(const std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y)
{
  std::ifstream file(filename);
  std::vector<std::vector<double>> data;
  std::string line;
  size_t cols = 0;

  // skip the header line
  if (std::getline(file, line))
  {
    // We don't really need to do anything with the header line, yet
    // Though we could parse it if needed to get column names
  }
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ',')) // Split by comma
    {
      row.push_back(std::stod(cell)); // Convert string to double
    }
    if (cols == 0)
      cols = row.size();
    data.push_back(row); // Add the row to the data vector
  }

  size_t nRows = data.size();
  size_t nCols = cols;
  X.resize(nRows, nCols - 1); // Features are all columns except the first
  Y.resize(nRows);            // Labels are the first column

  for (size_t i = 0; i < nRows; ++i)
  {
    Y(i) = data[i][0]; // Fill labels with the first column
    for (size_t j = 1; j < nCols; ++j)
    {
      X(i, j - 1) = data[i][j]; // Fill features, skipping the first column
    }
  }
}

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
FlexNN::splitXY(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, const std::vector<double> &proportions)
{
  size_t nRows = X.rows();
  std::vector<size_t> indices(nRows);
  std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., nRows-1

  // Shuffle indices
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // Calculate split sizes
  std::vector<size_t> sizes;
  size_t total = 0;
  for (size_t i = 0; i < proportions.size(); ++i)
  {
    size_t sz = static_cast<size_t>(proportions[i] * nRows); // Calculate size for this split
    sizes.push_back(sz);
    total += sz;
  }
  // Adjust last split to cover all rows (in case of rounding)
  if (!sizes.empty())
    sizes.back() += nRows - total;

  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splits;
  size_t start = 0;
  for (size_t k = 0; k < sizes.size(); ++k) // Iterate over each split size
  {
    size_t sz = sizes[k];
    Eigen::MatrixXd X_split(sz, X.cols()); // Create a new matrix for the split
    Eigen::VectorXd Y_split(sz);
    for (size_t i = 0; i < sz; ++i) // Fill the split matrices
    {
      X_split.row(i) = X.row(indices[start + i]);
      Y_split(i) = Y(indices[start + i]);
    }
    splits.emplace_back(std::move(X_split), std::move(Y_split)); // Add the split to the result vector
    start += sz;
  }
  return splits;
}