// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Classify raw float32 p-value words before arithmetic can flush a negative subnormal to zero. @internal */
export const MULTIPLE_TESTING_P_VALUE_WGSL = /* wgsl */ `
// Tested: positive float32 bit patterns from +0 through 1, or negative zero exactly.
fn isTestedBits(bits: u32) -> bool {
  return bits <= 0x3f800000u || bits == 0x80000000u;
}`;
