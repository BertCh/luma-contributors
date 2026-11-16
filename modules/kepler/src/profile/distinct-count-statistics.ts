// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';

/** Default exact set budget, retained from the dataset profiler. */
export const KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT = 10_000;
/** Explicit refinement is bounded to keep each exact set within a few megabytes. */
export const KEPLER_PROFILE_MAXIMUM_REFINED_DISTINCT_COUNT = 100_000;
/** 128 KiB bitmap: display estimates only; occupied bits give a deterministic lower bound. */
const DISTINCT_BITMAP_BIT_COUNT = 1 << 20;

export type KeplerDistinctCountStatistics = {
  count: number;
  exact: boolean;
  lowerBound: number;
  upperBound: number;
};

/**
 * Request an exact set through one past each decision threshold, within the refinement budget.
 * Above the budget, bounds can still settle a threshold; otherwise the requirement stays unknown.
 */
export function getKeplerDistinctCountExactLimit(thresholds: readonly number[] = []): number {
  let exactLimit = KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT;
  for (const threshold of thresholds) {
    validateCapacity(
      'KeplerDistinctCounter',
      'refinementThreshold',
      threshold,
      0,
      Number.MAX_SAFE_INTEGER
    );
    exactLimit = Math.max(
      exactLimit,
      Math.min(threshold + 1, KEPLER_PROFILE_MAXIMUM_REFINED_DISTINCT_COUNT)
    );
  }
  return exactLimit;
}

const hashScratchFloat = new Float64Array(1);
const hashScratchWords = new Uint32Array(hashScratchFloat.buffer);

/** Murmur3 32-bit finaliser: every input bit changes about half of the output bits. */
function mixHash(value: number): number {
  let hash = value >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return hash >>> 0;
}

function hashValue(value: number | string): number {
  if (typeof value === 'number') {
    // `value + 0` turns -0 into 0 so that both count as one value, like a Set does.
    hashScratchFloat[0] = value + 0;
    return mixHash(hashScratchWords[0] ^ mixHash(hashScratchWords[1] + 0x9e3779b9));
  }
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return mixHash(hash);
}

/**
 * Bounded-memory counter of accepted numbers or strings (callers filter nulls and invalid values).
 * Exact through the set budget, then linear counting for display. No probabilistic interval
 * authorizes capacity: each occupied bit witnesses at least one distinct value, and the exact
 * prefix witnesses at least budget + 1. The upper bound is the number of observations minus
 * repeats proved in that prefix. Collisions and even full bitmap saturation preserve both bounds.
 */
export class KeplerDistinctCounter {
  private exactValues: Set<number | string> | null = new Set();
  private bitmap: Uint32Array | null = null;
  private observationCount = 0;
  private knownRepeatCount = 0;
  private occupiedBitCount = 0;

  constructor(
    private readonly exactCountLimit = KEPLER_PROFILE_MAXIMUM_EXACT_DISTINCT_COUNT,
    private readonly bitmapBitCount = DISTINCT_BITMAP_BIT_COUNT
  ) {
    validateCapacity(
      'KeplerDistinctCounter',
      'exactCountLimit',
      exactCountLimit,
      1,
      KEPLER_PROFILE_MAXIMUM_REFINED_DISTINCT_COUNT
    );
    validateCapacity(
      'KeplerDistinctCounter',
      'bitmapBitCount',
      bitmapBitCount,
      32,
      DISTINCT_BITMAP_BIT_COUNT
    );
    validateRequirement(
      'KeplerDistinctCounter',
      (bitmapBitCount & (bitmapBitCount - 1)) === 0,
      'bitmapBitCount must be a power of two'
    );
  }

  add(value: number | string): void {
    this.observationCount++;
    if (this.exactValues) {
      const previousSize = this.exactValues.size;
      this.exactValues.add(value);
      if (this.exactValues.size === previousSize) this.knownRepeatCount++;
      if (this.exactValues.size > this.exactCountLimit) this.spill();
      return;
    }
    this.setBit(hashValue(value));
  }

  isExact(): boolean {
    return this.exactValues !== null;
  }

  getCount(): number {
    return this.getStatistics().count;
  }

  getStatistics(): KeplerDistinctCountStatistics {
    if (this.exactValues) {
      const count = this.exactValues.size;
      return {count, exact: true, lowerBound: count, upperBound: count};
    }
    const lowerBound = Math.max(this.exactCountLimit + 1, this.occupiedBitCount);
    const upperBound = this.observationCount - this.knownRepeatCount;
    const clearBitCount = this.bitmapBitCount - this.occupiedBitCount;
    // Saturation carries no usable estimate. The observation bound remains finite and safe.
    const estimate =
      clearBitCount === 0
        ? upperBound
        : Math.round(-this.bitmapBitCount * Math.log(clearBitCount / this.bitmapBitCount));
    return {
      count: Math.max(lowerBound, Math.min(estimate, upperBound)),
      exact: false,
      lowerBound,
      upperBound
    };
  }

  private spill(): void {
    this.bitmap = new Uint32Array(this.bitmapBitCount >>> 5);
    for (const value of this.exactValues as Set<number | string>) this.setBit(hashValue(value));
    this.exactValues = null;
  }

  private setBit(hash: number): void {
    const bitIndex = hash & (this.bitmapBitCount - 1);
    const wordIndex = bitIndex >>> 5;
    const bitMask = 1 << (bitIndex & 31);
    const bitmap = this.bitmap as Uint32Array;
    if ((bitmap[wordIndex] & bitMask) === 0) {
      bitmap[wordIndex] |= bitMask;
      this.occupiedBitCount++;
    }
  }
}
