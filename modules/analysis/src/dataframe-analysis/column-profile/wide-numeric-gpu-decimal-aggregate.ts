// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {DataType, Vector, type Decimal} from 'apache-arrow';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  combineDecimal128SumAccumulators,
  joinWordsToDecimal128,
  normalizeDecimal128Comparison,
  splitDecimal128ToWords,
  type Decimal128CompareOperator,
  type Decimal128Summary
} from './wide-numeric-gpu-decimal-reference';
import {runWideNumericKernel} from './wide-numeric-gpu-runner';

/** Result words: minimum words 3..0 (biased), maximum words 3..0 (biased), count, null count, 4 sums, 4 carries, negative count. */
const DECIMAL_RESULT_WORDS = 19;
const SIGN_BIAS = 0x80000000;

const DECIMAL_HELPER_SOURCE = `
fn decimalWords(wordOffset: u32) -> vec4<u32> {
  return vec4<u32>(words[wordOffset], words[wordOffset + 1u], words[wordOffset + 2u], words[wordOffset + 3u]);
}
fn decimalLess(first: vec4<u32>, second: vec4<u32>) -> bool {
  let aHigh = first.w ^ 0x80000000u;
  let bHigh = second.w ^ 0x80000000u;
  if (aHigh != bHigh) {return aHigh < bHigh;}
  if (first.z != second.z) {return first.z < second.z;}
  if (first.y != second.y) {return first.y < second.y;}
  return first.x < second.x;
}`;

function formatVec4(words: readonly number[]): string {
  return `vec4<u32>(${words.map(word => `${word >>> 0}u`).join(', ')})`;
}

/** WGSL boolean over the row for `column operator constant` after exact scale normalization. */
export function makeDecimal128SelectionExpression(props: {
  operator: Decimal128CompareOperator;
  constant: bigint;
  constantScale: number;
  columnScale: number;
}): string {
  const comparison = normalizeDecimal128Comparison(
    props.operator,
    props.constant,
    props.constantScale,
    props.columnScale
  );
  if (comparison.kind === 'constant') return comparison.result ? 'true' : 'false';
  const threshold = formatVec4(splitDecimal128ToWords(comparison.threshold));
  const value = 'decimalWords(wordOffset)';
  switch (comparison.operator) {
    case '<':
      return `decimalLess(${value}, ${threshold})`;
    case '<=':
      return `!decimalLess(${threshold}, ${value})`;
    case '>':
      return `decimalLess(${threshold}, ${value})`;
    case '>=':
      return `!decimalLess(${value}, ${threshold})`;
    case '=':
      return `all(${value} == ${threshold})`;
    default:
      return `any(${value} != ${threshold})`;
  }
}

/**
 * Exact count, null count, minimum, maximum and sum of a resident Decimal128 column (four uint32
 * words per row, every chunk scanned, only 76 bytes read back). The signed minimum and maximum
 * resolve word by word from the sign word down; the sum accumulates each word modulo 2^32 with
 * atomic carry counters (exact for fewer than 2^32 rows) and subtracts 2^128 per negative value.
 * An optional comparison against a decimal constant restricts the rows, which makes `count` an exact
 * compare-and-count and every other statistic a filtered aggregate.
 */
export async function aggregateDecimal128OnGpu(props: {
  device: Device;
  words: GPUVector<'uint32x4'>;
  scale: number;
  validity?: GPUVector<'uint32'>;
  comparison?: {operator: Decimal128CompareOperator; constant: bigint; constantScale: number};
}): Promise<Decimal128Summary> {
  const selectionExpression = props.comparison
    ? makeDecimal128SelectionExpression({...props.comparison, columnScale: props.scale})
    : undefined;
  const guard = `${selectionExpression ? 'if (!selected) {return;}' : ''}`;
  const matchMinimum = (depth: number) =>
    Array.from(
      {length: depth},
      (_, index) =>
        `(${index === 0 ? 'wordBiased3' : `words[wordOffset + ${3 - index}u]`} == atomicLoad(&result[${index}]))`
    ).join(' && ');
  const matchMaximum = (depth: number) =>
    Array.from(
      {length: depth},
      (_, index) =>
        `(${index === 0 ? 'wordBiased3' : `words[wordOffset + ${3 - index}u]`} == atomicLoad(&result[${4 + index}]))`
    ).join(' && ');
  const wordPhase = (depth: number) => `${guard}
  if (!present) {return;}
  let wordBiased3 = words[wordOffset + 3u] ^ ${SIGN_BIAS}u;
  if (${matchMinimum(depth)}) {atomicMin(&result[${depth}], words[wordOffset + ${3 - depth}u]);}
  if (${matchMaximum(depth)}) {atomicMax(&result[${4 + depth}], words[wordOffset + ${3 - depth}u]);}`;
  const initialWords = new Uint32Array(DECIMAL_RESULT_WORDS);
  initialWords.fill(0xffffffff, 0, 4);
  const result = await runWideNumericKernel({
    device: props.device,
    id: 'wide-numeric-decimal128-aggregate',
    words: props.words,
    wordsPerRow: 4,
    validity: props.validity,
    helperSource: DECIMAL_HELPER_SOURCE,
    selectionExpression,
    resultWordCount: DECIMAL_RESULT_WORDS,
    initialWords,
    phaseBodies: [
      `${guard}
  if (!present) {atomicAdd(&result[9], 1u); return;}
  atomicAdd(&result[8], 1u);
  let wordBiased3 = words[wordOffset + 3u] ^ ${SIGN_BIAS}u;
  atomicMin(&result[0], wordBiased3);
  atomicMax(&result[4], wordBiased3);
  for (var wordIndex = 0u; wordIndex < 4u; wordIndex++) {
    let value = words[wordOffset + wordIndex];
    let old = atomicAdd(&result[10u + wordIndex], value);
    if (old + value < old) {atomicAdd(&result[14u + wordIndex], 1u);}
  }
  if ((words[wordOffset + 3u] & 0x80000000u) != 0u) {atomicAdd(&result[18], 1u);}`,
      wordPhase(1),
      wordPhase(2),
      wordPhase(3)
    ]
  });
  const count = result[8];
  const decode = (offset: number) =>
    joinWordsToDecimal128([
      result[offset + 3],
      result[offset + 2],
      result[offset + 1],
      (result[offset] ^ SIGN_BIAS) >>> 0
    ]);
  return {
    count,
    nullCount: result[9],
    minimum: count ? decode(0) : null,
    maximum: count ? decode(4) : null,
    sum: combineDecimal128SumAccumulators(
      [result[10], result[11], result[12], result[13]],
      [result[14], result[15], result[16], result[17]],
      result[18]
    ),
    scale: props.scale
  };
}

/**
 * Uploads only Arrow Decimal128 words, preserving chunks. Every chunk must have bitWidth 128 and
 * the column's scale. Allocates validity for every chunk only when any chunk has nulls.
 * The returned destroy() releases all owned vectors.
 */
export function makeDecimal128WordVectors(
  device: Device,
  column: Vector<Decimal>
): {
  words: GPUVector<'uint32x4'>;
  validity: GPUVector<'uint32'> | undefined;
  destroy(): void;
} {
  validateRequirement(
    'makeDecimal128WordVectors',
    DataType.isDecimal(column.type) && column.type.bitWidth === 128,
    `column must be Decimal128 (bitWidth 128); received ${column.type} (bitWidth ${'bitWidth' in column.type ? column.type.bitWidth : 'none'})`
  );
  for (const [chunkIndex, chunk] of column.data.entries()) {
    validateRequirement(
      'makeDecimal128WordVectors',
      DataType.isDecimal(chunk.type) &&
        chunk.type.bitWidth === 128 &&
        chunk.type.scale === column.type.scale,
      `column chunk ${chunkIndex} must be Decimal128 (bitWidth 128) with scale ${column.type.scale}; received ${chunk.type} (bitWidth ${'bitWidth' in chunk.type ? chunk.type.bitWidth : 'none'}, scale ${'scale' in chunk.type ? chunk.type.scale : 'none'})`
    );
  }
  const wordData: GPUData<'uint32x4'>[] = [];
  const validityData: GPUData<'uint32'>[] = [];
  const allocatedBuffers: Buffer[] = [];
  const hasNulls = column.data.some(chunk => chunk.nullCount > 0);
  try {
    for (const chunk of column.data) {
      const values = chunk.values as unknown as Uint32Array;
      // Arrow has already sliced values; chunk.offset still applies to its validity bitmap.
      const words = values.subarray(0, chunk.length * 4);
      const wordBuffer = device.createBuffer({
        data: words.length ? words : new Uint32Array(4),
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
      allocatedBuffers.push(wordBuffer);
      wordData.push(
        new GPUData({
          format: 'uint32x4',
          length: chunk.length,
          ownsBuffer: true,
          buffer: wordBuffer
        })
      );
      if (!hasNulls) continue;
      const flags = new Uint32Array(Math.max(1, chunk.length)).fill(1);
      if (chunk.nullCount > 0) {
        const vector = new Vector([chunk]);
        for (let row = 0; row < chunk.length; row++) flags[row] = vector.get(row) === null ? 0 : 1;
      }
      const validityBuffer = device.createBuffer({
        data: flags,
        usage: Buffer.STORAGE | Buffer.COPY_DST
      });
      allocatedBuffers.push(validityBuffer);
      validityData.push(
        new GPUData({
          format: 'uint32',
          length: chunk.length,
          ownsBuffer: true,
          buffer: validityBuffer
        })
      );
    }
    const make = <Format extends 'uint32x4' | 'uint32'>(format: Format, data: GPUData<Format>[]) =>
      new GPUVector({name: `decimal128-${format}`, type: 'data', format, ownsData: true, data});
    const words = make('uint32x4', wordData);
    const validity = hasNulls ? make('uint32', validityData) : undefined;
    return {
      words,
      validity,
      destroy() {
        words.destroy();
        validity?.destroy();
      }
    };
  } catch (error) {
    // A failed factory returns no owner, so release even buffers allocated before a constructor failed.
    for (const buffer of allocatedBuffers) buffer.destroy();
    throw error;
  }
}

/** Uploads, aggregates and frees one Arrow Decimal128 column. */
export async function aggregateDecimal128Column(
  device: Device,
  column: Vector<Decimal>,
  comparison?: {operator: Decimal128CompareOperator; constant: bigint; constantScale: number}
): Promise<Decimal128Summary> {
  const vectors = makeDecimal128WordVectors(device, column);
  try {
    return await aggregateDecimal128OnGpu({
      device,
      words: vectors.words,
      scale: column.type.scale,
      validity: vectors.validity,
      comparison
    });
  } finally {
    vectors.destroy();
  }
}
