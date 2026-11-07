// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const MASK64 = (1n << 64n) - 1n;

export function makeWideNumericTestValues(): bigint[] {
  const edges = [
    0n,
    1n,
    2n,
    0xffn,
    0xffffn,
    0xffffffn,
    0x1000000n,
    0x1000001n,
    0x1000002n,
    0x1000003n,
    0x7fffffffn,
    0x80000000n,
    0xffffffffn,
    0x100000000n,
    0x100000001n,
    0x1fffffffffffffn,
    0x20000000000001n,
    0x7fffffffffffffffn,
    0x8000000000000000n,
    0x8000000000000001n,
    0xffffffffffffffffn,
    0xfffffffffffffffen,
    0x123456789abcdef0n,
    0xffffff7fffffffffn,
    0x0000ff8000000000n,
    0x0000ff8000000001n,
    0x0000ffc000000000n
  ];
  // Deterministic xorshift values complete the edge set.
  let state = 0x9e3779b97f4a7c15n;
  const generated: bigint[] = [];
  for (let index = 0; index < 200; index++) {
    state ^= (state << 13n) & MASK64;
    state ^= state >> 7n;
    state ^= (state << 17n) & MASK64;
    generated.push(state & (MASK64 >> BigInt(index % 64)));
  }
  return [...edges, ...generated];
}
