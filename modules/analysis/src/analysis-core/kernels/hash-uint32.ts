// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `hashUint32(value: u32) -> u32`: the lowbias32 integer hash (Chris Wellons,
 * https://nullprogram.com/blog/2018/07/31/). One definition for every kernel that needs a
 * stateless integer mix; {@link hashUint32} is the bit-identical TypeScript twin.
 *
 * @internal
 */
export const HASH_UINT32_WGSL = /* wgsl */ `
fn hashUint32(value: u32) -> u32 {
  var x = value;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}
`;

/** lowbias32 integer hash, bit-identical to the WGSL {@link HASH_UINT32_WGSL}. @internal */
export function hashUint32(value: number): number {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}
