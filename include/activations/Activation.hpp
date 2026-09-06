/**
 * @file Activation.hpp
 * @brief Public activation enum and helpers for FlexNN.
 *
 * Defines the type-safe `FlexNN::Activations::Activation` enum that replaces
 * the old stringly-typed `std::string activationFunction`. The enum is 1 byte
 * (uint8_t) so it maps directly to the `model.bin` flat binary (see
 * LLD_FLEXNN §7 / Appendix A). Per-activation elementwise ops live in
 * `include/activations` (e.g., ReLU.hpp) and are re-exported via
 * `FlexNN::Activations::detail` for use by `lib/layers`.
 *
 * Why this file exists:
 * - Gives `-Wswitch-enum` exhaustiveness: adding a new Activation forces
 *   updates to every switch without a `default` branch.
 * - No heap, no linear string compare in hot forward path.
 * - Keeps the public enum separate from the per-activation math so tests
 *   can include a single header (`Activation.hpp`) without pulling all
 *   per-activation headers unless needed.
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#include <Eigen/Dense>

#include "ActivationParameters.hpp"

namespace FlexNN::Activations {

/**
 * @brief Activation function identifier.
 *
 * Values are stable on-wire (model.bin) — do not reorder. 0 is linear
 * (pre-BatchNorm or debugging). LeakyReLU alpha is configurable per-layer
 * via `ActivationParameters` (default 0.01, see `ActivationParameters.hpp`);
 * `model.bin` v2 stores it as 1 float32 in `aux` when `act==LeakyReLU`.
 */
enum class Activation : uint8_t {
  None = 0,      ///< Linear, no activation
  ReLU = 1,      ///< max(0, z)
  LeakyReLU = 2, ///< z>0 ? z : alpha*z (alpha from ActivationParameters)
  Sigmoid = 3,   ///< 1/(1+exp(-z)), clamped
  Tanh = 4,      ///< std::tanh(z)
  Softmax = 5    ///< Stable col-wise softmax — only on last layer in v0.1
};
static_assert(sizeof(Activation) == 1, "Activation must be 1 byte for model.bin encoding");
static_assert(std::is_same_v<std::underlying_type_t<Activation>, uint8_t>,
              "Activation underlying type must be uint8_t");

/**
 * @brief Convert Activation to canonical lower-case string.
 *
 * @param a Activation value
 * @return Non-owning C-string (e.g., "relu", "leaky_relu"). Returns "none"
 *         for unknown values (should be unreachable — all 6 values are covered).
 */
[[nodiscard]] const char* to_string(Activation a) noexcept;

/**
 * @brief Parse a string into an Activation.
 *
 * Case-insensitive, trims ASCII whitespace. Accepts canonical names
 * ("relu", "leaky_relu", "leakyrelu", "sigmoid", "tanh", "softmax", "none")
 * and common aliases ("leaky-relu", "lrelu", "linear" for None). Returns
 * false on unknown input without throwing — caller decides whether to treat
 * as `None` or error.
 *
 * @note noexcept: heap allocation for lowercasing may throw bad_alloc and
 *       terminate via std::terminate (acceptable for host training; tests
 *       use small strings).
 *
 * @param s Input view (not null-terminated required)
 * @param out Output activation on success
 * @return true if parsed, false otherwise (out unchanged on false)
 */
[[nodiscard]] bool try_parse(std::string_view s, Activation& out) noexcept;

namespace detail {

/**
 * @brief Elementwise forward for a single activation.
 *
 * Pure function on `MatrixXd` (double). Used by `Layers::<Concrete>::forward()`
 * via `switch(act)`. Header-only per-activation impls are in
 * `include/activations/<Act>.hpp` — this dispatcher keeps layer code short.
 * `params` is only used when `act==LeakyReLU` (others ignore it).
 *
 * @param act Activation to apply
 * @param Z Pre-activation (linear output)
 * @param params Hyperparameters (e.g., leakyAlpha for LeakyReLU)
 * @return Post-activation matrix A, same shape as Z
 */
Eigen::MatrixXd forward(Activation act, const Eigen::MatrixXd& Z,
                        const ActivationParameters& params = ActivationParameters{});

/**
 * @brief Elementwise backward for a single activation (hidden layers).
 *
 * Computes `dZ = f'(Z) ⊙ dA` where `dA` is upstream gradient. For Softmax
 * hidden this is **not** fused — use `softmax_backward_hidden` instead.
 * For the last-layer Softmax + cross-entropy, the caller (`NeuralNetwork::backward`)
 * fuses and computes `dZ = (A - Y)/m` directly, never calling this.
 *
 * @note noexcept: allocates MatrixXd; bad_alloc terminates via std::terminate
 *       (acceptable for host; no heap in MCU runtime which does not use this).
 *
 * @param act Activation that was used in forward
 * @param dA Upstream gradient (same shape as Z/A)
 * @param Z Pre-activation from forward
 * @param A Post-activation from forward (needed for Sigmoid/Tanh to avoid recompute)
 * @param params Same params as forward (leakyAlpha for LeakyReLU)
 * @return Gradient w.r.t. Z, same shape as Z
 */
Eigen::MatrixXd backward(Activation act, const Eigen::MatrixXd& dA,
                         const Eigen::MatrixXd& Z, const Eigen::MatrixXd& A,
                         const ActivationParameters& params = ActivationParameters{}) noexcept;

/**
 * @brief Stable column-wise softmax forward.
 *
 * Implements `exp(Z - max(Z, col)) / sum(exp(...))` per column for
 * numerical stability. Used by `detail::forward(Softmax, Z)` and directly
 * in tests.
 */
Eigen::MatrixXd softmax_forward(const Eigen::MatrixXd& Z);

/**
 * @brief Softmax hidden backward: `dZ = J * upstream` per column.
 *
 * For hidden Softmax (disallowed in v0.1 but tested), the Jacobian is
 * `J = diag(s) - s s^T` where `s = softmax(Z)` per column. This helper
 * applies `J` to the upstream gradient column-wise. Last-layer Softmax
 * never calls this — it uses the fused `A - Y` path.
 *
 * @note noexcept: allocates MatrixXd; bad_alloc terminates.
 *
 * @param s Softmax output per column (from forward)
 * @param upstream Upstream gradient (W_next^T * dZ_next) per column
 * @return Gradient w.r.t. Z
 */
Eigen::MatrixXd softmax_backward_hidden(const Eigen::MatrixXd& s,
                                       const Eigen::MatrixXd& upstream) noexcept;

} // namespace detail

} // namespace FlexNN::Activations
