#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "BasicNN.h"
#include "Utility.h"

int main()
{
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    std::cout << "Reading CSV file..." << std::endl;
    BasicNN::readCSV_XY("data/mnist-digit-recognition.csv", X, Y);
    X = X.array() / 255.0; // Normalize the input data
    
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> data = BasicNN::splitXY(X, Y, {0.9, 0.1}); // Split data into training and test sets
    X = data[0].first; // Training set features
    Y = data[0].second; // Training set labels
    Eigen::MatrixXd X_test = data[1].first; // Test set features
    Eigen::VectorXd Y_test = data[1].second; // Test set labels
    
    std::cout << "Data loaded successfully." << std::endl;
    std::cout << "Training data size: " << X.rows() << " samples, " << X.cols() << " features." << std::endl;
    std::cout << "Test data size: " << X_test.rows() << " samples, " << X_test.cols() << " features." << std::endl;
    X.transposeInPlace(); // Now X is (features, samples)
    X_test.transposeInPlace(); // Now X_test is (features, samples)

    BasicNN::NeuralNetwork nn({
        BasicNN::Layer(X.rows(), 10, "relu"),
        BasicNN::Layer(10, 10, "softmax")
    });
    std::cout << "Neural Network created with 2 layers." << std::endl;


    std::cout << "Training started." << std::endl;
    nn.train(X, Y, 0.5, 300);
    std::cout << "Training completed." << std::endl;

    std::cout << "Accuracy on training data: " << nn.accuracy(X, Y) * 100 << "%" << std::endl;
    std::cout << "Accuracy on testing data: " << nn.accuracy(X_test, Y_test) * 100 << "%" << std::endl;
    

    std::cout << ">> ";
    int testIndex; std::cin >> testIndex; // Wait for user input to proceed with testing 
    while(testIndex){
        if (testIndex < 0 || testIndex >= X_test.cols()) {
            std::cout << "Invalid index. Please enter a number between 0 and " << X_test.cols() - 1 << "." << std::endl;
            std::cout << ">> ";
            std::cin >> testIndex; // Wait for user input to proceed with testing
            continue;
        }
        Eigen::VectorXd prediction = nn.predict(X_test.col(testIndex));
        int predictedClass;
        prediction.maxCoeff(&predictedClass); // Get the index of the maximum value in the prediction vector
        std::cout << "Predicted Label: " << predictedClass << std::endl;
        std::cout << "Actual Label: " << Y_test(testIndex) << std::endl;
        // Display the image as ASCII art
        const Eigen::VectorXd& img = X_test.col(testIndex);
        std::cout << "Image:" << std::endl;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                double pixel = img(i * 28 + j) * 255.0; // If normalized, multiply back
                char c;
                if (pixel > 200) c = '#';
                else if (pixel > 120) c = '*';
                else if (pixel > 50) c = '.';
                else c = ' ';
                std::cout << c;
            }
            std::cout << std::endl;
        }
        std::cout << ">> ";
        std::cin >> testIndex; // Wait for user input to proceed with the next test
    }
    return 0;
}