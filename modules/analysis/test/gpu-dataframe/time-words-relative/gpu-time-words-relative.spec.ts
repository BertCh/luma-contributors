// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/index';
import {GPUTimeWordsRelative} from '../../../src/dataframe-analysis/time-words-relative/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const INT64_MINIMUM = -(2n ** 63n);
const INT64_MAXIMUM = 2n ** 63n - 1n;

async function convert(
  ticks: readonly bigint[],
  temporalContext: GPUTemporalContext,
  origin: bigint
): Promise<{relative: number[]; context: GPUTemporalContext}> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'time-words-relative'});
  const timestamps = createInputBuffer(device, getInt64TimeWords(BigInt64Array.from(ticks)));
  const output = createOutputBuffer(device, ticks.length);
  const node = new GPUTimeWordsRelative({
    timestamps: importGraphBuffer(graph, 'times', timestamps, 'uint32x2', ticks.length),
    temporalContext,
    origin,
    output: importGraphBuffer(graph, 'relative', output, 'float32', ticks.length)
  });
  graph.add(node);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const relative = await readFloat32(output, ticks.length);
  compiled.destroy();
  timestamps.destroy();
  output.destroy();
  return {relative, context: node.outputTemporalContext};
}

function getContext(unit: GPUTemporalContext['unit'], epoch: GPUTemporalContext['epoch'] = 'unix') {
  return {unit, epoch, encoding: 'int64-words', axis: {kind: 'irregular'}} as GPUTemporalContext;
}

describe('GPUTimeWordsRelative on the GPU', () => {
  it('is exact within 2^24 ticks of the origin, before and after it and across 1970', async () => {
    const origin = 1_700_000_000_000n;
    const offsets = [
      0n,
      1n,
      -1n,
      999n,
      -1000n,
      16_777_216n,
      -16_777_216n,
      (86_400_000n * 100n) % 16_777_216n
    ];
    const ticks = offsets.map(offset => origin + offset);
    const {relative, context} = await convert(ticks, getContext('millisecond'), origin);
    expect(relative).toEqual(offsets.map(offset => Number(offset)));
    expect(context).toEqual({
      ...getContext('millisecond'),
      encoding: 'float32-relative',
      epoch: origin
    });

    // Origin before 1970 and times that straddle it.
    const early = BigInt(Date.UTC(1969, 11, 31, 23, 59, 0));
    const earlyTicks = [early - 61_000n, early, early + 60_000n, early + 61_001n, 0n];
    const earlyResult = await convert(earlyTicks, getContext('millisecond'), early);
    expect(earlyResult.relative).toEqual(earlyTicks.map(tick => Number(tick - early)));
  });

  it('subtracts on the words, so an absolute nanosecond time keeps its exact offset', async () => {
    // 1.7e18 ns needs 61 bits; as f32 it would round to 2^37 ns steps, but the offset to a nearby
    // origin is exact.
    const origin = 1_700_000_000_123_456_789n;
    const offsets = [0n, 1n, -1n, 1_000n, -123_456n, 5_000_000n];
    const {relative} = await convert(
      offsets.map(offset => origin + offset),
      getContext('nanosecond'),
      origin
    );
    expect(relative).toEqual(offsets.map(Number));
    expect(Math.fround(Number(origin))).not.toBe(Number(origin));
  });

  it('rounds large differences to float32 with the right sign, up to the Int64 extremes', async () => {
    const origin = 0n;
    const ticks = [INT64_MAXIMUM, INT64_MINIMUM, 3n * 2n ** 40n + 12_345n, -(5n * 2n ** 33n) - 7n];
    const {relative} = await convert(ticks, getContext('second'), origin);
    ticks.forEach((tick, index) => {
      const expected = Number(tick);
      expect(Math.sign(relative[index])).toBe(Math.sign(expected));
      // Two roundings through f32 halves: relative error below 2^-22.
      expect(Math.abs(relative[index] - expected) / Math.abs(expected)).toBeLessThan(2 ** -22);
    });
    // A pair further apart than Int64 still gets the right sign: |INT64_MAX - INT64_MIN| < 2^64.
    const wide = await convert([INT64_MAXIMUM], getContext('second'), INT64_MINIMUM);
    expect(wide.relative[0] / 2 ** 64).toBeCloseTo(1, 5);
  });

  it('counts from the origin even when the words count from another epoch', async () => {
    // Words are ticks after 1,000; the origin 5,000 is the word 4,000.
    const {relative, context} = await convert(
      [4_000n, 4_010n, 3_990n],
      getContext('millisecond', 1_000n),
      5_000n
    );
    expect(relative).toEqual([0, 10, -10]);
    expect(context.epoch).toBe(5_000n);
  });
});
