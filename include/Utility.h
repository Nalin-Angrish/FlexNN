#ifndef BasicNN_UTILITY_H
#define BasicNN_UTILITY_H

#include <Eigen/Dense>
#include <vector>

namespace BasicNN
{
  Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd &Y, int num_classes)
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
}

#endif // BasicNN_UTILITY_H
