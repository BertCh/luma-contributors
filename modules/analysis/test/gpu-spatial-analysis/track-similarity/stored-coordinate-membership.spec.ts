// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createDynamicProgramHarness} from './track-similarity-dtw-frechet-harness';
import {runMeasures, runSearch} from './track-similarity-edit-measures-harness';

it('TWN-05 decides stored coordinates in direct, named and search GPU forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Exact distances: 1+2^-25 > 1; sqrt(1+2^-26) > 1; 1 = 1.
  for (const [firstX, secondY, within] of [
    [-(2 ** -25), 0, 0],
    [0, 2 ** -13, 0],
    [0, 0, 1]
  ]) {
    const first = [[[firstX, 0] as const]];
    const second = [[[1, secondY] as const]];
    const direct = createDynamicProgramHarness(device, first, second);
    try {
      const result = await direct.run(1);
      expect(Array.from(result.frechet)).toEqual([1]);
      expect(Array.from(result.within)).toEqual([within]);
      expect(Array.from(result.status)).toEqual([0]);
    } finally {
      direct.destroy();
    }
    const edits = await runMeasures(device, first, second, [
      'editDistanceOnRealSequence',
      'longestCommonSubsequence'
    ]);
    expect(Array.from(edits.editDistanceOnRealSequence)).toEqual([1 - within]);
    expect(Array.from(edits.longestCommonSubsequence)).toEqual([within]);
    expect(Array.from(edits.status)).toEqual([0]);
    const named = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(firstX, 0, 1, secondY)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        first: {format: 'uint32', values: Uint32Array.of(0)},
        second: {format: 'uint32', values: Uint32Array.of(1)},
        epsilon: {format: 'float32', values: Float32Array.of(1)}
      },
      operations: [
        new GPUTrackSimilarityOperation({
          linesA: {positions: 'positions', lineOffsets: 'offsets'},
          pairA: 'first',
          pairB: 'second',
          pairCount: 1,
          frechetEpsilon: 'epsilon',
          output: {
            frechet: 'distance',
            frechetWithin: 'within',
            editDistanceOnRealSequence: 'edits',
            longestCommonSubsequence: 'similarity',
            pairStatus: 'status'
          }
        })
      ],
      outputs: ['distance', 'within', 'edits', 'similarity', 'status']
    });
    for (const [name, expected] of Object.entries({
      distance: 1,
      within,
      edits: 1 - within,
      similarity: within,
      status: 0
    })) {
      expect(Array.from(named[name])).toEqual([expected]);
    }
    for (const metric of [
      'frechet',
      'edit-distance-on-real-sequence',
      'longest-common-subsequence'
    ] as const) {
      const result = await runSearch(device, first, second, Uint32Array.of(0), Uint32Array.of(0), {
        metric,
        threshold: metric === 'edit-distance-on-real-sequence' ? 0 : 1,
        topK: 1
      });
      expect(Array.from(result.matched)).toEqual([within]);
      // LCSS rejects impossible matches at its bound; the distance searches evaluate these pairs.
      expect(Array.from(result.status)).toEqual([
        metric === 'longest-common-subsequence' && !within ? 1 : 0
      ]);
      expect(Array.from(result.distance)).toEqual([
        metric === 'longest-common-subsequence'
          ? within
            ? 1
            : NaN
          : metric === 'frechet'
            ? 1
            : 1 - within
      ]);
    }
  }
});

it('S1 exposes the documented float32 virtual-midpoint difference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.of(1048576, 0, 1048576.125, 4, 1048576.125, 0, 1048576.25, 4)
      },
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 4)},
      first: {format: 'uint32', values: Uint32Array.of(0)},
      second: {format: 'uint32', values: Uint32Array.of(1)},
      epsilon: {format: 'float32', values: Float32Array.of(0.125)}
    },
    operations: [
      new GPUTrackSimilarityOperation({
        linesA: {positions: 'positions', lineOffsets: 'offsets'},
        pairA: 'first',
        pairB: 'second',
        pairCount: 1,
        densify: 0.5,
        editDistanceTolerance: 0.125,
        lcssTolerance: 0.125,
        frechetEpsilon: 'epsilon',
        output: {
          frechetWithin: 'within',
          editDistanceOnRealSequence: 'edits',
          longestCommonSubsequence: 'similarity',
          pairStatus: 'status'
        }
      })
    ],
    outputs: ['within', 'edits', 'similarity', 'status']
  });
  // Only the two endpoint matches remain; the rounded virtual midpoints differ by 1/4 > 1/8.
  expect(Array.from(result['within'])).toEqual([0]);
  expect(Array.from(result['edits'])).toEqual([1]);
  expect(Array.from(result['similarity'])).toEqual([Math.fround(2 / 3)]);
  expect(Array.from(result['status'])).toEqual([0]);
});

it('S2 preserves tiny normal-coordinate membership on devices that flush subnormal differences', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const radius = 2 ** -126;
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.of(radius, radius, 1.75 * radius, 1.75 * radius)
      },
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
      first: {format: 'uint32', values: Uint32Array.of(0)},
      second: {format: 'uint32', values: Uint32Array.of(1)},
      epsilon: {format: 'float32', values: Float32Array.of(radius)}
    },
    operations: [
      new GPUTrackSimilarityOperation({
        linesA: {positions: 'positions', lineOffsets: 'offsets'},
        pairA: 'first',
        pairB: 'second',
        pairCount: 1,
        editDistanceTolerance: radius,
        lcssTolerance: radius,
        frechetEpsilon: 'epsilon',
        output: {
          frechetWithin: 'within',
          editDistanceOnRealSequence: 'edits',
          longestCommonSubsequence: 'similarity',
          pairStatus: 'status'
        }
      })
    ],
    outputs: ['within', 'edits', 'similarity', 'status']
  });
  // Exact squared separation is 9/8 times radius².
  expect(Array.from(result['within'])).toEqual([0]);
  expect(Array.from(result['edits'])).toEqual([1]);
  expect(Array.from(result['similarity'])).toEqual([0]);
  expect(Array.from(result['status'])).toEqual([0]);
});
