// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export function getRoadClass(featureClass: string): number {
  if (/motorway|trunk/.test(featureClass)) return 0;
  if (/primary|secondary/.test(featureClass)) return 1;
  if (/tertiary/.test(featureClass)) return 2;
  return 3;
}

/** Mulberry32: small deterministic PRNG returning values in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic per-point hash in [0, 1) used to thin the mask. */
export function hashToUnit(index: number): number {
  let hash = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}
