#ifndef BasicNN_UTILITY_H
#define BasicNN_UTILITY_H

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace BasicNN
{
  Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd &Y, int num_classes);

  // Reads a CSV file and splits into X (all columns except first) and Y (first column)
  void readCSV_XY(const std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y);

  // Splits X and Y into multiple sets according to proportions.
  // proportions: e.g. {0.7, 0.2, 0.1} for train/val/test
  // Returns a vector of pairs: { {X1, Y1}, {X2, Y2}, ... }
  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>
  splitXY(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y, const std::vector<double> &proportions);
}

#endif // BasicNN_UTILITY_H