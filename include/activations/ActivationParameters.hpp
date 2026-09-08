/**
 * @file ActivationParameters.hpp
 * @brief Per-activation hyperparameters for FlexNN::Activations.
 *
 * Centralizes all activation-specific defaults so layers do not grow a
 * new member per activation (avoids `Dense::leakyAlpha`, `Dense::eluAlpha`, ...).
 * `ActivationParameters` is a trivial POD (8 B for v0.1) stored by-value in
 * concrete layers (`Dense`, `Conv1D`) alongside the `Activation` enum.
 * Layers pass it to `detail::forward/backward`; non-Leaky activations ignore it.
 *
 * Future extensions (ELU alpha, Softmax temperature, etc.) add a field here
 * with a constexpr default and update `ModelIO` aux handling without touching
 * every layer's ctor signature.
 */

#pragma once

namespace FlexNN::Activations {

/// Default slope for LeakyReLU negative side (matches LLD_FLEXNN §4, v0.1).
inline constexpr double kDefaultLeakyAlpha = 0.01;

/**
 * @brief Hyperparameters for activations.
 *
 * Holds defaults for all activations. Only the field matching the active
 * `Activation` is used; others are ignored. `leakyAlpha` is validated to be
 * `0 < alpha < 1` when `act==LeakyReLU` (see `Activation.cpp` export check).
 */
struct ActivationParameters {
  double leakyAlpha = kDefaultLeakyAlpha; ///< slope for z<=0 when LeakyReLU

  constexpr bool operator==(const ActivationParameters& o) const noexcept {
    return leakyAlpha == o.leakyAlpha;
  }
  constexpr bool operator!=(const ActivationParameters& o) const noexcept {
    return !(*this == o);
  }
};

} // namespace FlexNN::Activations
