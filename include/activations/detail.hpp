/**
 * @file detail.hpp
 * @brief Convenience re-export for activation detail headers.
 *
 * Including this file pulls in all per-activation inline helpers. Prefer
 * including the specific `ReLU.hpp` etc. in new code if you only need one
 * activation — this file is for `Activation.cpp` and tests that need all.
 */

#pragma once

#include "None.hpp"
#include "ReLU.hpp"
#include "LeakyReLU.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "Softmax.hpp"
