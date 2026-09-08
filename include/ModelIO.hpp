/**
 * @file ModelIO.hpp
 * @brief Flat binary ModelIO for FlexNN — exportModel / importModel.
 *
 * Implements the 20B main header + 52B per-layer header + blobs + file CRC
 * contract from LLD_FLEXNN §7 / Appendix A. FlexNN writes, any runtime reads.
 * No TinyForge dependency — just `Layers::Layer` variant via `std::visit`.
 *
 * All integers little-endian, floats IEEE-754 binary32 little-endian, no
 * padding, CRC-32 IEEE (poly 0xEDB88320, init 0xFFFFFFFF, xorout 0xFFFFFFFF).
 */

#pragma once

#include <cstdint>
#include <string>

#include "FlexNN.h"

namespace FlexNN {

/**
 * @brief Result of export/import.
 *
 * `ok==true` means success and `error` is empty. `ok==false` means failure
 * and `error` contains a human-readable reason (e.g., "dilation !=1 unsupported",
 * "magic mismatch", "CRC mismatch").
 */
struct Status {
  bool ok = true;
  std::string error;

  static Status Ok() { return {true, ""}; }
  static Status Err(std::string msg) { return {false, std::move(msg)}; }
};

/**
 * @brief Options for export (currently version).
 *
 * v1: flat binary with enum activation only (LeakyReLU alpha fixed 0.01, no aux).
 * v2: adds per-layer LeakyReLU alpha as 1 float32 in `aux` when `act==LeakyReLU`
 *     (see `ActivationParameters`). Default is 2 so new code writes the
 *     configurable alpha (still 0.01 if user never changed it).
 */
struct ExportOptions {
  int formatVersion = 2; ///< 1=legacy, 2=with ActivationParameters
};

/**
 * @brief Export a trained NeuralNetwork to flat binary `model.bin`.
 *
 * Validates each layer (e.g., Conv1D dilation==1, Softmax only on last layer,
 * unknown DType, LeakyReLU alpha in (0,1)) and returns `Status::Err` on
 * impossible export without writing. Otherwise writes main header (20B) +
 * N*52B LayerHeaders + blobs (float32 LE, row-major) + file CRC32 and
 * returns `Ok`. v2 writes `ActivationParameters` for LeakyReLU as `aux`.
 *
 * @param net Network to export (weights are double, cast to float32)
 * @param path Destination file (truncated if exists)
 * @param opts Export options (formatVersion 1 or 2)
 * @return Status::Ok on success, Err with message on failure
 */
[[nodiscard]] Status exportModel(const NeuralNetwork& net, const std::string& path,
                                 ExportOptions opts = {});

/**
 * @brief Import a flat binary `model.bin` into a NeuralNetwork.
 *
 * Validates magic, version, header_len, layer_count, header/file CRC32,
 * offsets, and per-type counts. Reconstructs `Layers::Dense` / `Conv1D` with
 * the same activation and float32 blobs cast back to double.
 *
 * @param net Output network (overwritten on success, unchanged on failure)
 * @param path Source file
 * @return Status::Ok on success, Err on failure
 */
[[nodiscard]] Status importModel(NeuralNetwork& net, const std::string& path);

} // namespace FlexNN
