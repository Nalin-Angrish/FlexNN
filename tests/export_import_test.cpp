/**
 * @file export_import_test.cpp
 * @brief Round-trip tests for ModelIO flat binary (Dense + Conv1D).
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <vector>

#include "FlexNN.h"
#include "ModelIO.hpp"
#include "layers/Dense.hpp"
#include "layers/Conv1D.hpp"
#include "layers/BatchNorm1D.hpp"
#include "activations/Activation.hpp"

using namespace FlexNN;
using namespace FlexNN::Layers;
using namespace FlexNN::Activations;

// Helper to compare two networks' weights (tolerance for float32 round-trip)
static void expect_networks_near(const NeuralNetwork& a, const NeuralNetwork& b,
                                 double tol = 1e-6) {
  ASSERT_EQ(a.layers().size(), b.layers().size());
  for (size_t i = 0; i < a.layers().size(); ++i) {
    const auto& la = a.layers()[i];
    const auto& lb = b.layers()[i];
    EXPECT_EQ(la.type(), lb.type());
    EXPECT_EQ(la.activation(), lb.activation());
    // Compare ActivationParameters for LeakyReLU (tolerance for float32)
    if (la.activation() == Activation::LeakyReLU) {
      EXPECT_NEAR(la.activationParams().leakyAlpha, lb.activationParams().leakyAlpha, 1e-6)
          << "leakyAlpha mismatch at layer " << i;
    }
    std::visit(
        [&](auto&& va) {
          using T = std::decay_t<decltype(va)>;
          const auto* vb = std::get_if<T>(&lb.variant());
          ASSERT_NE(vb, nullptr);
          if constexpr (std::is_same_v<T, Dense> || std::is_same_v<T, Conv1D>) {
            EXPECT_TRUE(va.weights().isApprox(vb->weights(), tol))
                << "weights mismatch at layer " << i;
            EXPECT_TRUE(va.biases().isApprox(vb->biases(), tol))
                << "biases mismatch at layer " << i;
          } else if constexpr (std::is_same_v<T, BatchNorm1D>) {
            EXPECT_TRUE(va.gamma().isApprox(vb->gamma(), tol))
                << "gamma mismatch at layer " << i;
            EXPECT_TRUE(va.beta().isApprox(vb->beta(), tol))
                << "beta mismatch at layer " << i;
            EXPECT_TRUE(va.runningMean().isApprox(vb->runningMean(), tol))
                << "mean mismatch at layer " << i;
            EXPECT_TRUE(va.runningVar().isApprox(vb->runningVar(), tol))
                << "var mismatch at layer " << i;
          }
        },
        la.variant());
  }
}

TEST(ModelIO, DenseRoundTrip) {
  NeuralNetwork net(std::vector<Layers::Layer>{
      Dense(4, 3, Activation::ReLU),
      Dense(3, 2, Activation::Softmax),
  });
  const std::string path = "/tmp/flexnn_dense_roundtrip.bin";
  auto st = exportModel(net, path);
  ASSERT_TRUE(st.ok) << st.error;
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  auto st2 = importModel(net2, path);
  ASSERT_TRUE(st2.ok) << st2.error;
  expect_networks_near(net, net2);
  std::remove(path.c_str());
}

TEST(ModelIO, Conv1DRoundTrip) {
  NeuralNetwork net(std::vector<Layers::Layer>{
      Conv1D(Conv1DParams{2, 3, 3, 1, 1, 1}, Activation::ReLU),
      Dense(3 * 6, 2, Activation::Softmax), // L_in 6 -> L_out 6
  });
  const std::string path = "/tmp/flexnn_conv_roundtrip.bin";
  auto st = exportModel(net, path);
  ASSERT_TRUE(st.ok) << st.error;
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  auto st2 = importModel(net2, path);
  ASSERT_TRUE(st2.ok) << st2.error;
  expect_networks_near(net, net2);
  std::remove(path.c_str());
}

TEST(ModelIO, MixedDenseConvRoundTrip) {
  NeuralNetwork net(std::vector<Layers::Layer>{
      Dense(4, 6, Activation::Tanh),
      Conv1D(Conv1DParams{2, 2, 3, 1, 1, 1}, Activation::Sigmoid),
      Dense(2 * 6, 3, Activation::Softmax),
  });
  const std::string path = "/tmp/flexnn_mixed_roundtrip.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  ASSERT_TRUE(importModel(net2, path).ok);
  expect_networks_near(net, net2);
  std::remove(path.c_str());
}

TEST(ModelIO, HeaderAndCrc) {
  NeuralNetwork net(std::vector<Layers::Layer>{Dense(2, 2, Activation::None)});
  const std::string path = "/tmp/flexnn_header.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  // Read file and check magic/version (default export is v2 with ActivationParameters)
  std::ifstream in(path, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  ASSERT_GE(data.size(), 20u + 52u + 4u);
  uint32_t magic = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
  EXPECT_EQ(magic, 0x54464E47u);
  uint16_t version = data[4] | (data[5] << 8);
  EXPECT_EQ(version, 2);
  uint16_t header_len = data[6] | (data[7] << 8);
  EXPECT_EQ(header_len, 20);
  uint32_t layer_count = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24);
  EXPECT_EQ(layer_count, 1u);
  std::remove(path.c_str());
}

TEST(ModelIO, NegativeCorruptMagic) {
  NeuralNetwork net(std::vector<Layers::Layer>{Dense(2, 2, Activation::ReLU)});
  const std::string path = "/tmp/flexnn_corrupt_magic.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  // Corrupt magic — any corruption should cause import to fail (file CRC or magic)
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  f.seekp(0);
  uint8_t bad = 0xFF;
  f.write(reinterpret_cast<char*>(&bad), 1);
  f.close();
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  auto st = importModel(net2, path);
  EXPECT_FALSE(st.ok);
  EXPECT_FALSE(st.error.empty());
  std::remove(path.c_str());
}

TEST(ModelIO, NegativeBumpVersion) {
  NeuralNetwork net(std::vector<Layers::Layer>{Dense(2, 2, Activation::ReLU)});
  const std::string path = "/tmp/flexnn_bump_version.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  // Bump version to 3 and fix header CRC to make it look like a newer file,
  // but importer should still reject version !=1/2
  // For simplicity, just overwrite version bytes and don't fix CRC — importer will
  // fail on header CRC first, which is also a valid rejection. To test version,
  // we need to fix CRC.
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  // version at offset 4
  f.seekp(4);
  uint8_t v[2] = {3, 0};
  f.write(reinterpret_cast<char*>(v), 2);
  // Recompute header CRC over first 16B
  f.seekp(0);
  std::vector<uint8_t> hdr(16);
  f.read(reinterpret_cast<char*>(hdr.data()), 16);
  // Compute CRC (same as in ModelIO.cpp)
  auto crc32 = [](const uint8_t* data, size_t len) -> uint32_t {
    static const std::array<uint32_t, 256> table = [] {
      std::array<uint32_t, 256> t{};
      for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        t[i] = c;
      }
      return t;
    }();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
  };
  uint32_t new_crc = crc32(hdr.data(), 16);
  f.seekp(16);
  uint8_t crc_bytes[4] = {static_cast<uint8_t>(new_crc & 0xFF),
                         static_cast<uint8_t>((new_crc >> 8) & 0xFF),
                         static_cast<uint8_t>((new_crc >> 16) & 0xFF),
                         static_cast<uint8_t>((new_crc >> 24) & 0xFF)};
  f.write(reinterpret_cast<char*>(crc_bytes), 4);
  f.close();
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  auto st = importModel(net2, path);
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.error.find("version"), std::string::npos);
  std::remove(path.c_str());
}

TEST(ModelIO, NegativeTruncate) {
  NeuralNetwork net(std::vector<Layers::Layer>{Dense(2, 2, Activation::ReLU)});
  const std::string path = "/tmp/flexnn_truncate.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  // Truncate file by 10 bytes
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  size_t sz = static_cast<size_t>(in.tellg());
  in.close();
  // Truncate
  std::string cmd = "truncate -s " + std::to_string(sz - 10) + " " + path;
  int ret = system(cmd.c_str());
  (void)ret;
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  auto st = importModel(net2, path);
  EXPECT_FALSE(st.ok);
  std::remove(path.c_str());
}

TEST(ModelIO, NegativeDilation) {
  NeuralNetwork net(std::vector<Layers::Layer>{
      Conv1D(Conv1DParams{1, 1, 3, 1, 1, 2}, Activation::ReLU), // dilation 2
  });
  auto st = exportModel(net, "/tmp/flexnn_dilation.bin");
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.error.find("dilation"), std::string::npos);
}

TEST(ModelIO, NegativeSoftmaxNotLast) {
  NeuralNetwork net(std::vector<Layers::Layer>{
      Dense(2, 2, Activation::Softmax),
      Dense(2, 2, Activation::ReLU),
  });
  auto st = exportModel(net, "/tmp/flexnn_softmax.bin");
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.error.find("Softmax"), std::string::npos);
}

TEST(ModelIO, LeakyReLUDefaultRoundTrip) {
  NeuralNetwork net(std::vector<Layers::Layer>{
      Dense(4, 3, Activation::LeakyReLU),
      Dense(3, 2, Activation::Softmax),
  });
  const std::string path = "/tmp/flexnn_leaky_default.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  ASSERT_TRUE(importModel(net2, path).ok);
  expect_networks_near(net, net2);
  // Check aux was written (v2)
  std::ifstream in(path, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  uint16_t ver = data[4] | (data[5] << 8);
  EXPECT_EQ(ver, 2);
  std::remove(path.c_str());
}

TEST(ModelIO, LeakyReLUCustomAlphaRoundTrip) {
  ActivationParameters p;
  p.leakyAlpha = 0.2;
  NeuralNetwork net(std::vector<Layers::Layer>{
      Dense(4, 3, Activation::LeakyReLU, p),
      Conv1D(Conv1DParams{1, 2, 3, 1, 1, 1}, Activation::LeakyReLU, p),
      Dense(2 * 6, 2, Activation::Softmax),
  });
  const std::string path = "/tmp/flexnn_leaky_custom.bin";
  ASSERT_TRUE(exportModel(net, path).ok);
  NeuralNetwork net2(std::vector<Layers::Layer>{Dense(1, 1)});
  ASSERT_TRUE(importModel(net2, path).ok);
  expect_networks_near(net, net2);
  EXPECT_NEAR(net2.layers()[0].activationParams().leakyAlpha, 0.2, 1e-6);
  EXPECT_NEAR(net2.layers()[1].activationParams().leakyAlpha, 0.2, 1e-6);
  std::remove(path.c_str());
}

TEST(ModelIO, LeakyReLUCustomAlphaRequiresV2) {
  ActivationParameters p;
  p.leakyAlpha = 0.3;
  NeuralNetwork net(std::vector<Layers::Layer>{Dense(2, 2, Activation::LeakyReLU, p)});
  // Export with v1 should fail
  ExportOptions opts;
  opts.formatVersion = 1;
  auto st = exportModel(net, "/tmp/flexnn_leaky_v1.bin", opts);
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.error.find("formatVersion"), std::string::npos);
  // v2 should succeed
  opts.formatVersion = 2;
  EXPECT_TRUE(exportModel(net, "/tmp/flexnn_leaky_v2.bin", opts).ok);
  std::remove("/tmp/flexnn_leaky_v2.bin");
  std::remove("/tmp/flexnn_leaky_v1.bin");
}

TEST(ModelIO, LeakyReLUInvalidAlpha) {
  ActivationParameters p;
  p.leakyAlpha = 1.5; // invalid >1
  NeuralNetwork net(std::vector<Layers::Layer>{Dense(2, 2, Activation::LeakyReLU, p)});
  auto st = exportModel(net, "/tmp/flexnn_invalid.bin");
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.error.find("alpha"), std::string::npos);
}
