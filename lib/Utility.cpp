#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

#include "Utility.h"

Eigen::MatrixXd FlexNN::oneHotEncode(const Eigen::VectorXd &Y, int num_classes)
{
  Eigen::MatrixXd Y_onehot = Eigen::MatrixXd::Zero(num_classes, Y.size());
  for (int i = 0; i < Y.size(); ++i)
  {
    int label = static_cast<int>(Y(i));
    if (label >= 0 && label < num_classes)
      Y_onehot(label, i) = 1.0;
  }
  return Y_onehot;
}

// Reads a CSV file and splits into X (all columns except first) and Y (first column)
void FlexNN::readCSV_XY(const std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y)
{
  std::ifstream file(filename);
  std::vector<std::vector<double>> data;
  std::string line;
  size_t cols = 0;

  // skip the header line
  if (std::getline(file, line))
  {
    // Optionally process header if needed
  }
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ','))
    {
      row.push_back(std::stod(cell));
    }
    if (cols == 0)
      cols = row.size();
    data.push_back(row);
  }

  size_t nRows = data.size();
  size_t nCols = cols;
  X.resize(nRows, nCols - 1);
  Y.resize(nRows);

  for (size_t i = 0; i < nRows; ++i)
  {
    Y(i) = data[i][0];
    for (size_t j = 1; j < nCols; ++j)
    {
      X(i, j - 1) = data[i][j];
    }
  }
}

// Splits X and Y into multiple sets according to proportions.
// proportions: e.g. {0.7, 0.2, 0.1} for train/val/test
// Returns a vector of pairs: { {X1, Y1}, {X2, Y2}, ... }
std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>
FlexNN::splitXY(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, const std::vector<double> &proportions)
{
  size_t nRows = X.rows();
  std::vector<size_t> indices(nRows);
  std::iota(indices.begin(), indices.end(), 0);

  // Shuffle indices
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // Calculate split sizes
  std::vector<size_t> sizes;
  size_t total = 0;
  for (size_t i = 0; i < proportions.size(); ++i)
  {
    size_t sz = static_cast<size_t>(proportions[i] * nRows);
    sizes.push_back(sz);
    total += sz;
  }
  // Adjust last split to cover all rows (in case of rounding)
  if (!sizes.empty())
    sizes.back() += nRows - total;

  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> splits;
  size_t start = 0;
  for (size_t k = 0; k < sizes.size(); ++k)
  {
    size_t sz = sizes[k];
    Eigen::MatrixXd X_split(sz, X.cols());
    Eigen::VectorXd Y_split(sz);
    for (size_t i = 0; i < sz; ++i)
    {
      X_split.row(i) = X.row(indices[start + i]);
      Y_split(i) = Y(indices[start + i]);
    }
    splits.emplace_back(std::move(X_split), std::move(Y_split));
    start += sz;
  }
  return splits;
}