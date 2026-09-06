/**
 * @file Activation.cpp
 * @brief Implements FlexNN::Activations helpers (to_string, try_parse, dispatch).
 *
 * Why out-of-line: `to_string` and `try_parse` are non-trivial (string
 * handling) and not hot — keeping them in a .cpp avoids inlining bloat and
 * keeps `Activation.hpp` small for the many layer headers that include it.
 * The per-activation math stays header-only inline (see `include/activations`)
 * so the compiler can vectorize it at call sites.
 */

#include "activations/Activation.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "activations/detail.hpp"

namespace FlexNN::Activations {

// Exhaustive switch — no default so -Wswitch-enum will warn if a new
// Activation value is added but not handled here.
const char* to_string(Activation a) noexcept {
  switch (a) {
    case Activation::None:
      return "none";
    case Activation::ReLU:
      return "relu";
    case Activation::LeakyReLU:
      return "leaky_relu";
    case Activation::Sigmoid:
      return "sigmoid";
    case Activation::Tanh:
      return "tanh";
    case Activation::Softmax:
      return "softmax";
  }
  // Unreachable — all 6 values handled. Return "none" to silence -Wreturn-type.
  return "none";
}

bool try_parse(std::string_view s, Activation& out) noexcept {
  // Trim ASCII whitespace (space, tab, newline, CR)
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
    ++start;
  }
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  if (start >= end) {
    return false;
  }
  std::string lower;
  lower.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    // Use unsigned char to avoid UB on negative char values
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(s[i]))));
  }

  // Canonical names + common aliases (leaky variants).
  // We keep this mapping in one place so LLD_FLEXNN §4's `try_parse` contract
  // is implemented exactly once. Input is already lowercased, so only
  // lower-case comparisons are needed.
  if (lower == "none" || lower == "linear") {
    out = Activation::None;
    return true;
  }
  if (lower == "relu") {
    out = Activation::ReLU;
    return true;
  }
  if (lower == "leaky_relu" || lower == "leakyrelu" || lower == "leaky-relu" ||
      lower == "leaky_relu_0.01" || lower == "lrelu") {
    out = Activation::LeakyReLU;
    return true;
  }
  if (lower == "sigmoid") {
    out = Activation::Sigmoid;
    return true;
  }
  if (lower == "tanh") {
    out = Activation::Tanh;
    return true;
  }
  if (lower == "softmax") {
    out = Activation::Softmax;
    return true;
  }
  return false;
}

namespace detail {

Eigen::MatrixXd forward(Activation act, const Eigen::MatrixXd& Z) {
  // Exhaustive switch — no default so new Activation values trigger -Wswitch-enum.
  switch (act) {
    case Activation::None:
      return none_forward(Z);
    case Activation::ReLU:
      return relu_forward(Z);
    case Activation::LeakyReLU:
      return leaky_relu_forward(Z);
    case Activation::Sigmoid:
      return sigmoid_forward(Z);
    case Activation::Tanh:
      return tanh_forward(Z);
    case Activation::Softmax:
      return softmax_forward(Z);
  }
  return Z; // Unreachable
}

Eigen::MatrixXd backward(Activation act, const Eigen::MatrixXd& dA,
                         const Eigen::MatrixXd& Z, const Eigen::MatrixXd& A) noexcept {
  switch (act) {
    case Activation::None:
      return none_backward(dA, Z, A);
    case Activation::ReLU:
      return relu_backward(dA, Z, A);
    case Activation::LeakyReLU:
      return leaky_relu_backward(dA, Z, A);
    case Activation::Sigmoid:
      return sigmoid_backward(dA, Z, A);
    case Activation::Tanh:
      return tanh_backward(dA, Z, A);
    case Activation::Softmax:
      // Hidden Softmax uses Jacobian; last-layer Softmax is fused in
      // NeuralNetwork::backward and never reaches here. If it does, fall
      // back to hidden path (caller should have errored on export).
      return softmax_backward_hidden(A, dA);
  }
  return dA; // Unreachable
}

} // namespace detail

} // namespace FlexNN::Activations
