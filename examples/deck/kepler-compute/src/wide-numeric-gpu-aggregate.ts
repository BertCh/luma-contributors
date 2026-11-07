// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {combineWordSumAccumulators, type WideInteger64Summary} from './wide-numeric-gpu-reference';
import {runWideNumericKernel} from './wide-numeric-gpu-runner';

/** Result words: minimum high, minimum low, maximum high, maximum low, count, null count, sum accumulators, negative count. */
const EXACT_AGGREGATE_RESULT_WORDS = 11;
const SIGN_BIAS = 0x80000000;

/**
 * Exact count, null count, minimum, maximum and 128-bit-exact sum of a resident Int64 or Uint64
 * column over every chunk. Only 44 bytes are read back. The sum accumulates the low and high
 * words modulo 2^32 with atomic carry counters, so it is exact for up to 2^32 rows; a signed
 * sum subtracts 2^64 per negative value. Minimum and maximum use two phases (high word, then
 * low word among rows equal to the winning high word). Decimal128 stored as two Int64 words
 * per half can be aggregated by calling this once per half-view; combining halves is host work.
 */
export async function aggregateInteger64OnGpu(props: {
  device: Device;
  words: GPUVector<'uint32x2'>;
  signed: boolean;
  validity?: GPUVector<'uint32'>;
  /** WGSL boolean over `pair` restricting the rows (an unselected row counts as neither value nor null). */
  selectionExpression?: string;
}): Promise<WideInteger64Summary> {
  const bias = props.signed ? SIGN_BIAS : 0;
  const result = await runWideNumericKernel({
    device: props.device,
    id: 'wide-numeric-exact-aggregate',
    words: props.words,
    validity: props.validity,
    selectionExpression: props.selectionExpression,
    resultWordCount: EXACT_AGGREGATE_RESULT_WORDS,
    initialWords: Uint32Array.from([0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    phaseBodies: [
      `${props.selectionExpression ? 'if (!selected) {return;}' : ''}
  if (!present) {atomicAdd(&result[5], 1u); return;}
  atomicAdd(&result[4], 1u);
  let biasedHigh = pair.y ^ ${bias}u;
  atomicMin(&result[0], biasedHigh);
  atomicMax(&result[2], biasedHigh);
  let lowOld = atomicAdd(&result[6], pair.x);
  if (lowOld + pair.x < lowOld) {atomicAdd(&result[7], 1u);}
  let highOld = atomicAdd(&result[8], pair.y);
  if (highOld + pair.y < highOld) {atomicAdd(&result[9], 1u);}
  ${props.signed ? 'if ((pair.y & 0x80000000u) != 0u) {atomicAdd(&result[10], 1u);}' : ''}`,
      `if (!present || !selected) {return;}
  let biasedHigh = pair.y ^ ${bias}u;
  if (biasedHigh == atomicLoad(&result[0])) {atomicMin(&result[1], pair.x);}
  if (biasedHigh == atomicLoad(&result[2])) {atomicMax(&result[3], pair.x);}`
    ]
  });
  const decode = (highWord: number, lowWord: number) => {
    const bits = (BigInt((highWord ^ bias) >>> 0) << 32n) | BigInt(lowWord);
    return props.signed ? BigInt.asIntN(64, bits) : bits;
  };
  const count = result[4];
  return {
    count,
    nullCount: result[5],
    minimum: count ? decode(result[0], result[1]) : null,
    maximum: count ? decode(result[2], result[3]) : null,
    sum: combineWordSumAccumulators(result[6], result[7], result[8], result[9], result[10])
  };
}
