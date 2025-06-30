#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

#include "BasicNN.h"

// Reads a CSV file and splits into X (all columns except first) and Y (first column)
void readCSV_XY(const std::string& filename, Eigen::MatrixXd& X, Eigen::VectorXd& Y) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;
    size_t cols = 0;

    // skip the header line
    if (std::getline(file, line)) {
        // Optionally process header if needed
    }
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        if (cols == 0) cols = row.size();
        data.push_back(row);
    }

    size_t nRows = data.size();
    size_t nCols = cols;
    X.resize(nRows, nCols - 1);
    Y.resize(nRows);

    for (size_t i = 0; i < nRows; ++i) {
        Y(i) = data[i][0];
        for (size_t j = 1; j < nCols; ++j) {
            X(i, j - 1) = data[i][j];
        }
    }
}

Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd& Y, int num_classes) {
    Eigen::MatrixXd Y_onehot = Eigen::MatrixXd::Zero(num_classes, Y.size());
    for (int i = 0; i < Y.size(); ++i) {
        int label = static_cast<int>(Y(i));
        if (label >= 0 && label < num_classes)
            Y_onehot(label, i) = 1.0;
    }
    return Y_onehot;
}

int main()
{
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    std::cout << "Reading CSV file..." << std::endl;
    readCSV_XY("data/train.csv", X, Y);
    X.transposeInPlace(); // Now X is (features, samples)
    X = X.array() / 255.0; // Normalize the input data

    BasicNN::NeuralNetwork nn({
        BasicNN::Layer(X.rows(), 10, "relu"),
        BasicNN::Layer(10, 10, "softmax")
    });
    std::cout << "Neural Network created with 2 layers." << std::endl;


    std::cout << "Training started." << std::endl;
    nn.train(X, Y, 0.5, 500);
    std::cout << "Training completed." << std::endl;

    std::cout << "Accuracy: " << nn.accuracy(X, Y) * 100 << "%" << std::endl;
    return 0;
}