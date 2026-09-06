/**
 * @file LayerTypes.hpp
 * @brief Shared PODs for FlexNN::Layers — LayerType, DType, and per-type Params.
 *
 * This header is standalone (no Eigen) so both `ModelIO` and per-layer
 * headers can include it without pulling heavy dependencies. Values are
 * stable on-wire for `model.bin` (see LLD_FLEXNN §5, §7, Appendix A).
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace FlexNN::Layers {

/**
 * @brief Layer kind, stable on-wire (uint8_t).
 *
 * Do not reorder — `model.bin` stores this as 1 byte. New types are appended.
 */
enum class LayerType : uint8_t {
  Dense = 0,       ///< Fully-connected: W[out×in], b[out]
  Conv1D = 1,      ///< 1D temporal convolution
  BatchNorm1D = 2, ///< Per-feature BatchNorm (gamma/beta/mean/var in aux)
  MaxPool1D = 3,   ///< Max pooling, weightless
  AvgPool1D = 4    ///< Average pooling, weightless
};
static_assert(sizeof(LayerType) == 1, "LayerType must be 1 byte for model.bin");
static_assert(std::is_same_v<std::underlying_type_t<LayerType>, uint8_t>,
              "LayerType underlying type must be uint8_t");

/**
 * @brief Data type for blobs in `model.bin`.
 *
 * v0.1 only `F32` (float32 IEEE). Value 1 is reserved; importer must reject
 * unknown values (e.g., !=0) with Status error.
 */
enum class DType : uint16_t {
  F32 = 0 ///< float32 little-endian
};
static_assert(sizeof(DType) == 2, "DType must be 2 bytes for model.bin");
static_assert(std::is_same_v<std::underlying_type_t<DType>, uint16_t>,
              "DType underlying type must be uint16_t");

/**
 * @brief Parameters for a Dense layer.
 */
struct DenseParams {
  int inputSize = 0;  ///< Number of input features (rows of X)
  int outputSize = 0; ///< Number of output neurons (rows of Z/A)
};

/**
 * @brief Parameters for a Conv1D layer.
 *
 * Weight layout: `W[outChannels][inChannels*K]` row-major logical,
 * stored in Eigen column-major `MatrixXd` but serialized row-major
 * float32. Bias is `[outChannels]`.
 *
 * Output length: `L_out = (L_in + 2*pad - dilation*(K-1) -1)/stride +1`
 * Must be >=1 or export returns Status error.
 */
struct Conv1DParams {
  int inChannels = 0;
  int outChannels = 0;
  int kernelSize = 0;
  int stride = 1;
  int padding = 0;
  int dilation = 1; ///< Must be 1 in v0.1 (export error otherwise)
};

/**
 * @brief Parameters for pooling layers (Max and Avg share this POD).
 *
 * No weights. `channels` is both C_in and C_out (preserved).
 */
struct Pool1DParams {
  int channels = 0;
  int kernelSize = 0;
  int stride = 1;  ///< Default 1 to avoid div-by-zero in L_out
  int padding = 0;
};

/**
 * @brief Parameters for BatchNorm1D.
 *
 * Epsilon is fixed at 1e-5 in v0.1 and not serialized; momentum is also
 * fixed at 0.1 for running stats update. Future versions may serialize
 * epsilon via aux if needed.
 */
struct BatchNormParams {
  int numFeatures = 0;
  float epsilon = 1e-5f;
  float momentum = 0.1f;
};

} // namespace FlexNN::Layers
