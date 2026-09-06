/**
 * @file Layer.cpp
 * @brief Out-of-line helpers for FlexNN::Layers::Layer (currently header-only).
 *
 * This file exists so future non-inline helpers (e.g., ModelIO-specific
 * visit helpers, debug printing) have a home without growing the header.
 * Today the variant dispatch is header-only via `std::visit` in
 * `include/layers/Layer.hpp`, so this file is intentionally minimal.
 *
 * Keeping the file (even empty) ensures `CMakeLists.txt` already lists
 * `lib/layers/Layer.cpp` from PR-04, so later PRs that add Conv1D/BN/Pool
 * only need to edit the variant type list in the header.
 */

#include "layers/Layer.hpp"

// No out-of-line definitions needed in PR-04 — all dispatch is inline
// via std::visit in the header. This translation unit ensures the
// library still links when only Dense is active and gives a place for
// future helpers (e.g., `std::visit` pretty-printers).
