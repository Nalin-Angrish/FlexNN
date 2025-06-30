# My C++ Library

This project is a basic C++ library that demonstrates the use of CMake for building and managing a C++ project. It includes an entry point executable to test the library functionality.

## Project Structure

```
my-cpp-library
├── CMakeLists.txt        # CMake configuration file
├── README.md             # Project documentation
├── src                   # Source files
│   ├── main.cpp          # Entry point of the application
│   └── mylib.cpp         # Implementation of the library functions
├── include               # Header files
│   └── mylib.h           # Public API of the library
└── tests                 # Unit tests
    └── test_mylib.cpp    # Tests for the library functions
```

## Building the Project

To build the project, follow these steps:

1. Ensure you have CMake installed on your system.
2. Open a terminal and navigate to the project directory.
3. Create a build directory:
   ```
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```
   cmake ..
   ```
5. Build the project:
   ```
   cmake --build .
   ```

## Running the Executable

After building the project, you can run the entry point executable located in the `build` directory. This executable will test the functionality of the library.

## Running Tests

To run the unit tests, ensure you have a testing framework set up in your project. You can execute the tests from the build directory after building the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.