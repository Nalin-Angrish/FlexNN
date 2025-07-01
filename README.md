# FlexNN

Fully connected neural network built from scratch with flexible n-layer design and multiple activations.

## Example Usage

Below is a minimal example of how to use the FlexNN NeuralNetwork library in your own C++ project:

```cpp
#include <Eigen/Dense>
#include "FlexNN.h"

int main() {
    // Prepare your data (X: features, Y: labels)
    Eigen::MatrixXd X; // shape: (num_samples, num_features)
    Eigen::VectorXd Y; // shape: (num_samples)
    FlexNN::readCSV_XY("data/mnist-digit-recognition.csv", X, Y);

    // Normalize input data if needed
    X = X.array() / 255.0;

    // (Optional) Split data into training and test sets
    auto splits = FlexNN::splitXY(X, Y, {0.8, 0.2});
    Eigen::MatrixXd X_train = splits[0].first;
    Eigen::VectorXd Y_train = splits[0].second;
    Eigen::MatrixXd X_test = splits[1].first;
    Eigen::VectorXd Y_test = splits[1].second;

    // Transpose for (features, samples) shape expected by the network
    X_train.transposeInPlace();
    X_test.transposeInPlace();

    // Create a neural network with desired layers
    FlexNN::NeuralNetwork nn({
        FlexNN::Layer(X_train.rows(), 64, "relu"),
        FlexNN::Layer(64, 10, "softmax")
    });

    // Train the network
    nn.train(X_train, Y_train, 0.5, 100);

    // Evaluate accuracy
    double train_acc = nn.accuracy(X_train, Y_train);
    double test_acc = nn.accuracy(X_test, Y_test);

    std::cout << "Training accuracy: " << train_acc * 100 << "%" << std::endl;
    std::cout << "Test accuracy: " << test_acc * 100 << "%" << std::endl;

    return 0;
}
```

**Note:**  
- You can customize the network architecture by changing the number and type of layers.
- Make sure your data is in the correct format and normalized as needed.
- See the `src/main.cpp` file for a more complete example.


## Requirements
The C++ Neural Network library uses `Eigen3` for matrix operations and `OpenMP` for multithreading. The project uses `CMake` for building the code.

On Ubuntu, you can install these requirements using:
```
sudo apt install cmake libeigen3-dev
```

## Building and Running the Project

To build the project, follow these steps:

1. Open a terminal and navigate to the project directory.
2. Create a build directory:
   ```
   mkdir build
   cd build
   ```
3. Run CMake to configure the project:
   ```
   cmake ..
   ```
4. Build the project:
   ```
   cmake --build .
   ```
5. To run the program, go into the project root directory (so that dataset path is correctly resolved)
   ```
   cd ..
   ```
   And then run the `main` executable present inside the `build` folder by:
   ```
   ./build/main
   ```

# API Reference
For details on the code structure, available classes, and how to use FlexNN in your own projects, please visit the full documentation here: [https://docs.nalinangrish.me/FlexNN](https://docs.nalinangrish.me/FlexNN).

## Known Issues / Limitations

- Only CPU computation is supported (no GPU).
- No support for convolutional or recurrent layers.
- Training large models may be slow.
- Currently there is no support for saving model architecture and weights, so you need to train the model on every execution.

## License

This project is licensed under the Apache-2.0 License. See the LICENSE file for more details.