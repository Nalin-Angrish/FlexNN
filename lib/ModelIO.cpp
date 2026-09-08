/**
 * @file ModelIO.cpp
 * @brief Implements FlexNN flat binary export/import per LLD_FLEXNN §7.
 *
 * Why this file exists: FlexNN is the source of truth for training; any
 * runtime (including TinyForge) consumes `model.bin` without Eigen. The
 * writer is intentionally small (no JSON/Protobuf) and deterministic
 * (little-endian, IEEE floats, CRC).
 *
 * Layout (all LE, packed, no padding):
 *   Main header 20B: magic 0x54464E47, version 1, header_len 20, layer_count,
 *                    flags 0, header_crc32 (CRC of first 16B)
 *   Per-layer 52B: type, act, dtype, in_dim, out_dim, w_cnt, b_cnt, aux_cnt,
 *                  reserved, w_off, b_off, aux_off, c_in, c_out, k, stride, pad, dilation
 *   Blobs: w (float32), b (float32), aux (float32) at absolute offsets
 *   Footer: file_crc32 (CRC of whole file except last 4B)
 */

#include "ModelIO.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "layers/Layer.hpp"
#include "activations/Activation.hpp"

namespace FlexNN {

// ---------------------------------------------------------------------------
// Little-endian helpers (host is assumed LE on x86_64, but we still encode LE explicitly)
// ---------------------------------------------------------------------------

static void write_u16_le(std::ofstream& f, uint16_t v) {
  uint8_t b[2] = {static_cast<uint8_t>(v & 0xFF),
                  static_cast<uint8_t>((v >> 8) & 0xFF)};
  f.write(reinterpret_cast<char*>(b), 2);
}

static void write_u32_le(std::ofstream& f, uint32_t v) {
  uint8_t b[4] = {static_cast<uint8_t>(v & 0xFF),
                  static_cast<uint8_t>((v >> 8) & 0xFF),
                  static_cast<uint8_t>((v >> 16) & 0xFF),
                  static_cast<uint8_t>((v >> 24) & 0xFF)};
  f.write(reinterpret_cast<char*>(b), 4);
}

static void write_f32_le(std::ofstream& f, float v) {
  static_assert(sizeof(float) == 4, "float must be 32-bit");
  uint32_t iv;
  std::memcpy(&iv, &v, 4);
  write_u32_le(f, iv);
}

static uint16_t read_u16_le(const uint8_t* p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

static uint32_t read_u32_le(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static float read_f32_le(const uint8_t* p) {
  uint32_t iv = read_u32_le(p);
  float v;
  std::memcpy(&v, &iv, 4);
  return v;
}

// ---------------------------------------------------------------------------
// CRC-32 IEEE (poly 0xEDB88320, init 0xFFFFFFFF, xorout 0xFFFFFFFF, refin/refout true)
// Table generated via standard algorithm.
// ---------------------------------------------------------------------------

static uint32_t crc32_ieee(const uint8_t* data, size_t len,
                           uint32_t init = 0xFFFFFFFFu) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      t[i] = c;
    }
    return t;
  }();
  uint32_t crc = init;
  for (size_t i = 0; i < len; ++i) {
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Header structs (packed, 20B and 52B)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct MainHeader {
  uint32_t magic = 0x54464E47; // 'TFNG' LE
  uint16_t version = 1;
  uint16_t header_len = 20;
  uint32_t layer_count = 0;
  uint32_t flags = 0;
  uint32_t header_crc32 = 0; // CRC of first 16B
};
struct LayerHeader {
  uint8_t type = 0;
  uint8_t act = 0;
  uint16_t dtype = 0; // 0=F32
  uint32_t in_dim = 0;
  uint32_t out_dim = 0;
  uint32_t w_cnt = 0;
  uint32_t b_cnt = 0;
  uint32_t aux_cnt = 0;
  uint32_t reserved = 0;
  uint32_t w_off = 0;
  uint32_t b_off = 0;
  uint32_t aux_off = 0;
  uint16_t c_in = 0;
  uint16_t c_out = 0;
  uint16_t k = 0;
  uint16_t stride = 0;
  uint16_t pad = 0;
  uint16_t dilation = 0;
};
#pragma pack(pop)
static_assert(sizeof(MainHeader) == 20, "MainHeader must be 20B");
static_assert(sizeof(LayerHeader) == 52, "LayerHeader must be 52B");

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

Status exportModel(const NeuralNetwork& net, const std::string& path,
                   ExportOptions opts) {
  if (opts.formatVersion != 1 && opts.formatVersion != 2) {
    return Status::Err("unsupported formatVersion (only 1 or 2)");
  }

  const auto& layers = net.layers();
  size_t N = layers.size();
  if (N == 0) {
    return Status::Err("no layers to export");
  }
  if (N > 0xFFFF) {
    return Status::Err("too many layers");
  }
  // Validate each layer
  for (size_t i = 0; i < N; ++i) {
    const auto& l = layers[i];
    auto act = l.activation();
    auto type = l.type();
    // Softmax only on last layer in v0.1
    if (act == Activations::Activation::Softmax && i + 1 != N) {
      return Status::Err("Softmax only on last layer in v0.1");
    }
    // LeakyReLU alpha must be 0 < alpha < 1
    if (act == Activations::Activation::LeakyReLU) {
      double a = l.activationParams().leakyAlpha;
      if (!(a > 0.0 && a < 1.0)) {
        return Status::Err("LeakyReLU alpha must be 0 < alpha < 1");
      }
      if (opts.formatVersion == 1 && a != Activations::kDefaultLeakyAlpha) {
        return Status::Err("LeakyReLU custom alpha requires formatVersion 2");
      }
    }
    // Dilation check for Conv1D
    if (type == Layers::LayerType::Conv1D) {
      const auto* c = l.asConv1D();
      assert(c != nullptr);
      if (c->params().dilation != 1) {
        return Status::Err("Conv1D dilation !=1 unsupported in v0.1");
      }
      if (c->params().kernelSize <= 0 || c->params().inChannels <= 0 ||
          c->params().outChannels <= 0) {
        return Status::Err("Conv1D invalid dims");
      }
    }
    // DType only F32
    // (no other checks — unknown type/act still writes so future runtime can read)
  }

  // Build layer headers and compute offsets
  std::vector<LayerHeader> headers;
  headers.reserve(N);
  // Offsets start after main header + N*52
  uint32_t cur_off = 20 + static_cast<uint32_t>(N * 52);
  // We need to know blob sizes; for Conv1D in_dim/out_dim we set to 0 for now
  // (flattened sizes not needed for this PR — Dense uses in/out, Conv1D uses channels)
  for (size_t i = 0; i < N; ++i) {
    const auto& l = layers[i];
    LayerHeader h{};
    h.type = static_cast<uint8_t>(l.type());
    h.act = static_cast<uint8_t>(l.activation());
    h.dtype = static_cast<uint16_t>(Layers::DType::F32);
    // Per-type fields
    std::visit(
        [&](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Layers::Dense>) {
            h.in_dim = static_cast<uint32_t>(v.params().inputSize);
            h.out_dim = static_cast<uint32_t>(v.params().outputSize);
            h.w_cnt = static_cast<uint32_t>(v.weights().size());
            h.b_cnt = static_cast<uint32_t>(v.biases().size());
            h.aux_cnt = (v.activation() == Activations::Activation::LeakyReLU &&
                         opts.formatVersion == 2) ? 1 : 0;
            h.c_in = 0;
            h.c_out = 0;
            h.k = 0;
            h.stride = 0;
            h.pad = 0;
            h.dilation = 0;
          } else if constexpr (std::is_same_v<T, Layers::Conv1D>) {
            auto p = v.params();
            // For Conv1D, in_dim/out_dim are flattened (C*L) but L not stored;
            // we set to 0 and rely on c_in/c_out/k/stride/pad/dilation.
            // Future PRs may store L_in/L_out if needed; importer tolerates 0.
            h.in_dim = 0;
            h.out_dim = 0;
            h.w_cnt = static_cast<uint32_t>(v.weights().size());
            h.b_cnt = static_cast<uint32_t>(v.biases().size());
            h.aux_cnt = (v.activation() == Activations::Activation::LeakyReLU &&
                         opts.formatVersion == 2) ? 1 : 0;
            h.c_in = static_cast<uint16_t>(p.inChannels);
            h.c_out = static_cast<uint16_t>(p.outChannels);
            h.k = static_cast<uint16_t>(p.kernelSize);
            h.stride = static_cast<uint16_t>(p.stride);
            h.pad = static_cast<uint16_t>(p.padding);
            h.dilation = static_cast<uint16_t>(p.dilation);
          } else {
            // Future types (BN, Pool) will be handled in later PRs
            h.in_dim = 0;
            h.out_dim = 0;
          }
        },
        l.variant());

    // Offsets: tightly packed, no padding
    if (h.w_cnt > 0) {
      h.w_off = cur_off;
      cur_off += h.w_cnt * 4;
    } else {
      h.w_off = 0;
    }
    if (h.b_cnt > 0) {
      h.b_off = cur_off;
      cur_off += h.b_cnt * 4;
    } else {
      h.b_off = 0;
    }
    if (h.aux_cnt > 0) {
      h.aux_off = cur_off;
      cur_off += h.aux_cnt * 4;
    } else {
      h.aux_off = 0;
    }
    headers.push_back(h);
  }

  uint32_t file_size_without_crc = cur_off;
  // cur_off now points after last blob; file CRC will be 4B more

  // Open file
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return Status::Err("failed to open file for writing: " + path);
  }

  // Write main header (first 16B, then CRC)
  MainHeader mh{};
  mh.version = static_cast<uint16_t>(opts.formatVersion);
  mh.layer_count = static_cast<uint32_t>(N);
  // Compute header CRC over first 16B (magic..flags)
  uint8_t hdr_tmp[16];
  // Manually serialize first 16B for CRC
  hdr_tmp[0] = static_cast<uint8_t>(mh.magic & 0xFF);
  hdr_tmp[1] = static_cast<uint8_t>((mh.magic >> 8) & 0xFF);
  hdr_tmp[2] = static_cast<uint8_t>((mh.magic >> 16) & 0xFF);
  hdr_tmp[3] = static_cast<uint8_t>((mh.magic >> 24) & 0xFF);
  hdr_tmp[4] = static_cast<uint8_t>(mh.version & 0xFF);
  hdr_tmp[5] = static_cast<uint8_t>((mh.version >> 8) & 0xFF);
  hdr_tmp[6] = static_cast<uint8_t>(mh.header_len & 0xFF);
  hdr_tmp[7] = static_cast<uint8_t>((mh.header_len >> 8) & 0xFF);
  hdr_tmp[8] = static_cast<uint8_t>(mh.layer_count & 0xFF);
  hdr_tmp[9] = static_cast<uint8_t>((mh.layer_count >> 8) & 0xFF);
  hdr_tmp[10] = static_cast<uint8_t>((mh.layer_count >> 16) & 0xFF);
  hdr_tmp[11] = static_cast<uint8_t>((mh.layer_count >> 24) & 0xFF);
  hdr_tmp[12] = static_cast<uint8_t>(mh.flags & 0xFF);
  hdr_tmp[13] = static_cast<uint8_t>((mh.flags >> 8) & 0xFF);
  hdr_tmp[14] = static_cast<uint8_t>((mh.flags >> 16) & 0xFF);
  hdr_tmp[15] = static_cast<uint8_t>((mh.flags >> 24) & 0xFF);
  mh.header_crc32 = crc32_ieee(hdr_tmp, 16);

  // Serialize main header
  write_u32_le(out, mh.magic);
  write_u16_le(out, mh.version);
  write_u16_le(out, mh.header_len);
  write_u32_le(out, mh.layer_count);
  write_u32_le(out, mh.flags);
  write_u32_le(out, mh.header_crc32);

  // Serialize layer headers
  for (const auto& h : headers) {
    out.write(reinterpret_cast<const char*>(&h.type), 1);
    out.write(reinterpret_cast<const char*>(&h.act), 1);
    write_u16_le(out, h.dtype);
    write_u32_le(out, h.in_dim);
    write_u32_le(out, h.out_dim);
    write_u32_le(out, h.w_cnt);
    write_u32_le(out, h.b_cnt);
    write_u32_le(out, h.aux_cnt);
    write_u32_le(out, h.reserved);
    write_u32_le(out, h.w_off);
    write_u32_le(out, h.b_off);
    write_u32_le(out, h.aux_off);
    write_u16_le(out, h.c_in);
    write_u16_le(out, h.c_out);
    write_u16_le(out, h.k);
    write_u16_le(out, h.stride);
    write_u16_le(out, h.pad);
    write_u16_le(out, h.dilation);
  }

  // Serialize blobs (row-major float32 LE)
  // We need to collect all blobs in order of file offsets. Since headers
  // already computed w_off/b_off/aux_off in file order, we just write in that order
  // which is w,b,aux per layer sequentially as we computed.
  for (size_t i = 0; i < N; ++i) {
    const auto& l = layers[i];
    std::visit(
        [&](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Layers::Dense>) {
            // W row-major
            const auto& W = v.weights();
            for (Eigen::Index r = 0; r < W.rows(); ++r) {
              for (Eigen::Index c = 0; c < W.cols(); ++c) {
                float fv = static_cast<float>(W(r, c));
                write_f32_le(out, fv);
              }
            }
            const auto& b = v.biases();
            for (Eigen::Index r = 0; r < b.size(); ++r) {
              float fv = static_cast<float>(b(r));
              write_f32_le(out, fv);
            }
            // Aux for LeakyReLU alpha (v2 only, 1 float)
            if (v.activation() == Activations::Activation::LeakyReLU &&
                opts.formatVersion == 2) {
              write_f32_le(out, static_cast<float>(v.activationParams().leakyAlpha));
            }
          } else if constexpr (std::is_same_v<T, Layers::Conv1D>) {
            const auto& W = v.weights();
            for (Eigen::Index r = 0; r < W.rows(); ++r) {
              for (Eigen::Index c = 0; c < W.cols(); ++c) {
                float fv = static_cast<float>(W(r, c));
                write_f32_le(out, fv);
              }
            }
            const auto& b = v.biases();
            for (Eigen::Index r = 0; r < b.size(); ++r) {
              float fv = static_cast<float>(b(r));
              write_f32_le(out, fv);
            }
            if (v.activation() == Activations::Activation::LeakyReLU &&
                opts.formatVersion == 2) {
              write_f32_le(out, static_cast<float>(v.activationParams().leakyAlpha));
            }
          }
        },
        l.variant());
  }

  out.close();
  if (!out) {
    return Status::Err("failed to write file: " + path);
  }

  // Now compute file CRC (over entire file except last 4B) and append
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::Err("failed to reopen file for CRC: " + path);
  }
  std::vector<uint8_t> file_bytes(file_size_without_crc);
  in.read(reinterpret_cast<char*>(file_bytes.data()), file_size_without_crc);
  if (static_cast<size_t>(in.gcount()) != file_size_without_crc) {
    return Status::Err("failed to read back file for CRC");
  }
  in.close();
  uint32_t file_crc = crc32_ieee(file_bytes.data(), file_bytes.size());
  std::ofstream out2(path, std::ios::binary | std::ios::app);
  if (!out2) {
    return Status::Err("failed to reopen file for CRC append");
  }
  write_u32_le(out2, file_crc);
  out2.close();

  return Status::Ok();
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

Status importModel(NeuralNetwork& net, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::Err("failed to open file for reading: " + path);
  }
  // Read entire file
  in.seekg(0, std::ios::end);
  size_t file_size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  if (file_size < 20 + 4) {
    return Status::Err("file too small");
  }
  std::vector<uint8_t> data(file_size);
  in.read(reinterpret_cast<char*>(data.data()), file_size);
  if (static_cast<size_t>(in.gcount()) != file_size) {
    return Status::Err("failed to read file");
  }
  in.close();

  // Validate file CRC (last 4B)
  uint32_t stored_file_crc = read_u32_le(data.data() + file_size - 4);
  uint32_t computed_file_crc = crc32_ieee(data.data(), file_size - 4);
  if (stored_file_crc != computed_file_crc) {
    return Status::Err("file CRC mismatch");
  }

  // Parse main header
  if (file_size < 20) return Status::Err("file too small for main header");
  uint32_t magic = read_u32_le(data.data() + 0);
  if (magic != 0x54464E47) {
    return Status::Err("magic mismatch (expected TFNG)");
  }
  uint16_t version = read_u16_le(data.data() + 4);
  if (version != 1 && version != 2) {
    return Status::Err("unsupported version (only 1 or 2)");
  }
  uint16_t header_len = read_u16_le(data.data() + 6);
  if (header_len != 20) {
    return Status::Err("header_len !=20");
  }
  uint32_t layer_count = read_u32_le(data.data() + 8);
  if (layer_count == 0) {
    return Status::Err("no layers");
  }
  uint32_t flags = read_u32_le(data.data() + 12);
  if (flags != 0) {
    return Status::Err("flags must be 0");
  }
  uint32_t header_crc = read_u32_le(data.data() + 16);
  uint32_t computed_hdr_crc = crc32_ieee(data.data(), 16);
  if (header_crc != computed_hdr_crc) {
    return Status::Err("header CRC mismatch");
  }

  size_t expected_min = 20 + layer_count * 52 + 4;
  if (file_size < expected_min) {
    return Status::Err("file too small for layer headers");
  }

  // Parse layer headers
  std::vector<LayerHeader> headers;
  headers.reserve(layer_count);
  for (uint32_t i = 0; i < layer_count; ++i) {
    size_t off = 20 + i * 52;
    LayerHeader h{};
    h.type = data[off + 0];
    h.act = data[off + 1];
    h.dtype = read_u16_le(data.data() + off + 2);
    h.in_dim = read_u32_le(data.data() + off + 4);
    h.out_dim = read_u32_le(data.data() + off + 8);
    h.w_cnt = read_u32_le(data.data() + off + 12);
    h.b_cnt = read_u32_le(data.data() + off + 16);
    h.aux_cnt = read_u32_le(data.data() + off + 20);
    h.reserved = read_u32_le(data.data() + off + 24);
    h.w_off = read_u32_le(data.data() + off + 28);
    h.b_off = read_u32_le(data.data() + off + 32);
    h.aux_off = read_u32_le(data.data() + off + 36);
    h.c_in = read_u16_le(data.data() + off + 40);
    h.c_out = read_u16_le(data.data() + off + 42);
    h.k = read_u16_le(data.data() + off + 44);
    h.stride = read_u16_le(data.data() + off + 46);
    h.pad = read_u16_le(data.data() + off + 48);
    h.dilation = read_u16_le(data.data() + off + 50);

    if (h.dtype != 0) {
      return Status::Err("unsupported dtype (only F32)");
    }
    if (h.reserved != 0) {
      return Status::Err("reserved must be 0");
    }
    // Validate offsets (use size_t to avoid uint32_t overflow on cnt*4)
    auto check_off = [&](uint32_t cnt, uint32_t off) -> bool {
      if (cnt == 0) return off == 0;
      size_t end = static_cast<size_t>(off) + static_cast<size_t>(cnt) * 4;
      if (off < 20 + static_cast<size_t>(layer_count) * 52) return false;
      if (end > file_size - 4) return false;
      return true;
    };
    if (!check_off(h.w_cnt, h.w_off) || !check_off(h.b_cnt, h.b_off) ||
        !check_off(h.aux_cnt, h.aux_off)) {
      return Status::Err("invalid blob offset/count");
    }
    headers.push_back(h);
  }

  // Validate non-overlapping blobs across all layers (sorted intervals)
  {
    struct Interval {
      size_t start, end;
    };
    std::vector<Interval> intervals;
    intervals.reserve(layer_count * 3);
    for (const auto& h : headers) {
      if (h.w_cnt > 0) intervals.push_back({h.w_off, h.w_off + static_cast<size_t>(h.w_cnt) * 4});
      if (h.b_cnt > 0) intervals.push_back({h.b_off, h.b_off + static_cast<size_t>(h.b_cnt) * 4});
      if (h.aux_cnt > 0) intervals.push_back({h.aux_off, h.aux_off + static_cast<size_t>(h.aux_cnt) * 4});
    }
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& a, const Interval& b) { return a.start < b.start; });
    for (size_t i = 1; i < intervals.size(); ++i) {
      if (intervals[i].start < intervals[i - 1].end) {
        return Status::Err("overlapping blob offsets");
      }
    }
  }

  // Validate Softmax only on last layer (import must also enforce)
  for (uint32_t i = 0; i < layer_count; ++i) {
    if (headers[i].act == static_cast<uint8_t>(Activations::Activation::Softmax) &&
        i + 1 != layer_count) {
      return Status::Err("Softmax only on last layer");
    }
  }

  // Reconstruct network
  std::vector<Layers::Layer> new_layers;
  new_layers.reserve(layer_count);
  for (uint32_t i = 0; i < layer_count; ++i) {
    const auto& h = headers[i];
    Activations::Activation act = static_cast<Activations::Activation>(h.act);
    // Validate act range (0..5)
    if (h.act > 5) {
      return Status::Err("unknown activation");
    }
    if (h.type > 4) {
      return Status::Err("unknown layer type");
    }
    Layers::LayerType type = static_cast<Layers::LayerType>(h.type);
    if (type == Layers::LayerType::Dense) {
      // For Dense, in_dim/out_dim are authoritative; w_cnt must be in*out
      if (h.w_cnt != h.in_dim * h.out_dim) {
        return Status::Err("Dense w_cnt mismatch in_dim*out_dim");
      }
      if (h.b_cnt != h.out_dim) {
        return Status::Err("Dense b_cnt mismatch out_dim");
      }
      if (h.c_in != 0 || h.c_out != 0 || h.k != 0) {
        // For Dense, c_in/c_out/k should be 0
      }
      Activations::ActivationParameters denseActParams;
      if (act == Activations::Activation::LeakyReLU) {
        if (version == 1) {
          if (h.aux_cnt != 0) {
            return Status::Err("LeakyReLU aux_cnt must be 0 in v1");
          }
          denseActParams = Activations::ActivationParameters{};
        } else {
          if (h.aux_cnt != 1) {
            return Status::Err("LeakyReLU aux_cnt must be 1 in v2");
          }
          float fv = read_f32_le(data.data() + h.aux_off);
          double alpha = static_cast<double>(fv);
          if (!(alpha > 0.0 && alpha < 1.0)) {
            return Status::Err("LeakyReLU alpha must be 0 < alpha < 1");
          }
          denseActParams.leakyAlpha = alpha;
        }
      } else {
        if (h.aux_cnt != 0) {
          return Status::Err("non-LeakyReLU Dense aux_cnt must be 0");
        }
        denseActParams = Activations::ActivationParameters{};
      }
      Layers::DenseParams p{static_cast<int>(h.in_dim),
                            static_cast<int>(h.out_dim)};
      Layers::Dense d(p, act, denseActParams);
      // Load W row-major
      Eigen::MatrixXd W(h.out_dim, h.in_dim);
      for (uint32_t r = 0; r < h.out_dim; ++r) {
        for (uint32_t c = 0; c < h.in_dim; ++c) {
          size_t idx = h.w_off + (r * h.in_dim + c) * 4;
          float fv = read_f32_le(data.data() + idx);
          W(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
              static_cast<double>(fv);
        }
      }
      Eigen::VectorXd b(h.out_dim);
      for (uint32_t r = 0; r < h.out_dim; ++r) {
        size_t idx = h.b_off + r * 4;
        float fv = read_f32_le(data.data() + idx);
        b(static_cast<Eigen::Index>(r)) = static_cast<double>(fv);
      }
      d.setWeights(W);
      d.setBiases(b);
      new_layers.emplace_back(std::move(d));
    } else if (type == Layers::LayerType::Conv1D) {
      if (h.c_in == 0 || h.c_out == 0 || h.k == 0) {
        return Status::Err("Conv1D missing c_in/c_out/k");
      }
      uint32_t expected_w = static_cast<uint32_t>(h.c_out) * h.c_in * h.k;
      if (h.w_cnt != expected_w) {
        return Status::Err("Conv1D w_cnt mismatch c_out*c_in*k");
      }
      if (h.b_cnt != h.c_out) {
        return Status::Err("Conv1D b_cnt mismatch c_out");
      }
      if (h.dilation != 1) {
        return Status::Err("Conv1D dilation !=1 unsupported on import");
      }
      Activations::ActivationParameters convActParams;
      if (act == Activations::Activation::LeakyReLU) {
        if (version == 1) {
          if (h.aux_cnt != 0) {
            return Status::Err("LeakyReLU aux_cnt must be 0 in v1");
          }
          convActParams = Activations::ActivationParameters{};
        } else {
          if (h.aux_cnt != 1) {
            return Status::Err("LeakyReLU aux_cnt must be 1 in v2");
          }
          float fv = read_f32_le(data.data() + h.aux_off);
          double alpha = static_cast<double>(fv);
          if (!(alpha > 0.0 && alpha < 1.0)) {
            return Status::Err("LeakyReLU alpha must be 0 < alpha < 1");
          }
          convActParams.leakyAlpha = alpha;
        }
      } else {
        if (h.aux_cnt != 0) {
          return Status::Err("non-LeakyReLU Conv1D aux_cnt must be 0");
        }
        convActParams = Activations::ActivationParameters{};
      }
      Layers::Conv1DParams p{static_cast<int>(h.c_in),
                             static_cast<int>(h.c_out),
                             static_cast<int>(h.k),
                             static_cast<int>(h.stride),
                             static_cast<int>(h.pad),
                             static_cast<int>(h.dilation)};
      Layers::Conv1D c(p, act, convActParams);
      Eigen::MatrixXd W(h.c_out, h.c_in * h.k);
      for (uint32_t r = 0; r < h.c_out; ++r) {
        for (uint32_t cc = 0; cc < h.c_in * h.k; ++cc) {
          size_t idx = h.w_off + (r * h.c_in * h.k + cc) * 4;
          float fv = read_f32_le(data.data() + idx);
          W(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(cc)) =
              static_cast<double>(fv);
        }
      }
      Eigen::VectorXd b(h.c_out);
      for (uint32_t r = 0; r < h.c_out; ++r) {
        size_t idx = h.b_off + r * 4;
        float fv = read_f32_le(data.data() + idx);
        b(static_cast<Eigen::Index>(r)) = static_cast<double>(fv);
      }
      c.setWeights(W);
      c.setBiases(b);
      new_layers.emplace_back(std::move(c));
    } else {
      return Status::Err("unsupported layer type in v0.1 (only Dense/Conv1D)");
    }
  }

  net = NeuralNetwork(std::move(new_layers));
  return Status::Ok();
}

} // namespace FlexNN
