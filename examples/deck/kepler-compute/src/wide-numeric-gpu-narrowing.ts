// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {NumericNarrowingReport} from './analysis-numeric-diagnostics';
import {decodeNarrowingError, type WideNumericGpuFormat} from './wide-numeric-gpu-reference';
import {runWideNumericKernel} from './wide-numeric-gpu-runner';

/** Result words: error key (max), error top word (max), null, inexact, overflow, non-finite counts. */
const NARROWING_RESULT_WORDS = 6;
const ERROR_KEY_BIAS = 1100;

/**
 * Scans EVERY row of a resident Float64 / Int64 / Uint64 column (uint32x2 words, all chunks) and
 * returns the same NumericNarrowingReport semantic as diagnoseFloat32Narrowing: exact rounding
 * to the nearest-even Float32 decides inexact / overflow / non-finite per row; only six words
 * are read back. `maximumAbsoluteError` is reconstructed from the exact error exponent and the
 * top 32 error bits, so it is a lower bound within a relative 2^-31 of the true maximum.
 */
export async function scanWideNumericColumnOnGpu(props: {
  device: Device;
  words: GPUVector<'uint32x2'>;
  format: WideNumericGpuFormat;
  validity?: GPUVector<'uint32'>;
}): Promise<NumericNarrowingReport> {
  const classify =
    props.format === 'float64'
      ? 'classifyDoubleNarrowing(pair)'
      : `classifyIntegerNarrowing(pair, ${props.format === 'int64'})`;
  const result = await runWideNumericKernel({
    device: props.device,
    id: 'wide-numeric-narrowing-scan',
    words: props.words,
    validity: props.validity,
    resultWordCount: NARROWING_RESULT_WORDS,
    phaseBodies: [
      `if (!present) {atomicAdd(&result[2], 1u); return;}
  let narrowing = ${classify};
  if (narrowing.kind == 1u) {atomicAdd(&result[3], 1u); atomicMax(&result[0], narrowing.errorKey);}
  if (narrowing.kind == 2u) {atomicAdd(&result[4], 1u);}
  if (narrowing.kind == 3u) {atomicAdd(&result[5], 1u);}`,
      `if (!present) {return;}
  let narrowing = ${classify};
  if (narrowing.kind == 1u && narrowing.errorKey == atomicLoad(&result[0])) {atomicMax(&result[1], narrowing.errorTop);}`
    ]
  });
  const rowCount = props.words.data.reduce((total, chunk) => total + chunk.length, 0);
  const inexactCount = result[3];
  const overflowCount = result[4];
  return {
    format: props.format,
    rowCount,
    nullCount: result[2],
    inexactCount,
    overflowCount,
    nonFiniteCount: result[5],
    maximumAbsoluteError: inexactCount
      ? decodeNarrowingError(result[0] - ERROR_KEY_BIAS, result[1])
      : 0,
    verdict: overflowCount ? 'refused' : inexactCount ? 'lossy' : 'exact'
  };
}
