// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {
  GPUTrackSimilarityOperation,
  type TrackSimilarityOperationProps
} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {
  runOperationGraph,
  type OperationGraphInput,
  type OperationGraphRows
} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('computes maximum distance from original vertices beyond the unused dense-count limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      positionsA: {format: 'float32x2', values: new Float32Array(1048577 * 2)},
      offsetsA: {format: 'uint32', values: Uint32Array.of(0, 1048577)},
      positionsB: {format: 'float32x2', values: Float32Array.of(1, 0)},
      offsetsB: {format: 'uint32', values: Uint32Array.of(0, 1)},
      pairA: {format: 'uint32', values: Uint32Array.of(0)},
      pairB: {format: 'uint32', values: Uint32Array.of(0)}
    },
    operations: [
      new GPUTrackSimilarityOperation({
        linesA: {positions: 'positionsA', lineOffsets: 'offsetsA'},
        linesB: {positions: 'positionsB', lineOffsets: 'offsetsB'},
        pairA: 'pairA',
        pairB: 'pairB',
        pairCount: 1,
        densify: 1 / 4096,
        output: {maximumDistance: 'maximum', pairStatus: 'status'}
      })
    ],
    outputs: ['maximum', 'status']
  });
  expect(Array.from(result['status'])).toEqual([0]);
  expect(Array.from(result['maximum'])).toEqual([1]);
});

const OUTPUT = {
  hausdorff: 'hausdorff',
  frechet: 'frechet',
  frechetWithin: 'within',
  dynamicTimeWarping: 'warping',
  editDistanceOnRealSequence: 'edits',
  editDistanceWithRealPenalty: 'penalty',
  symmetricSegmentPathDistance: 'segmentPath',
  longestCommonSubsequence: 'subsequence',
  maximumDistance: 'maximum',
  pairStatus: 'status'
};

function compareOutputs(
  gpu: Record<string, OperationGraphRows>,
  cpu: Record<string, OperationGraphRows>
) {
  // EDR, status and decisions are integers: exact comparison. LCSS is a count/length ratio.
  // Distances and accumulated costs use relative tolerance 3e-5; zeros compare numerically.
  for (const name of Object.values(OUTPUT)) {
    expect(gpu[name].length).toBe(cpu[name].length);
    for (let index = 0; index < cpu[name].length; index++) {
      const expected = cpu[name][index];
      const actual = gpu[name][index];
      if (Number.isNaN(expected)) expect(Number.isNaN(actual), `${name}[${index}]`).toBe(true);
      else if (['edits', 'within', 'status'].includes(name) || expected === 0)
        expect(actual === expected, `${name}[${index}]`).toBe(true);
      else
        expect(Math.abs(actual - expected), `${name}[${index}]`).toBeLessThanOrEqual(
          3e-5 * Math.abs(expected)
        );
    }
  }
}

async function runComparison(
  inputs: Record<string, OperationGraphInput>,
  props: TrackSimilarityOperationProps
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUTrackSimilarityOperation(props)],
    outputs: Object.values(OUTPUT)
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUTrackSimilarityOperation(props)],
    outputs: Object.values(OUTPUT)
  });
  compareOutputs(gpu, cpu);
  return cpu;
}

it('compares every measure for multiple unequal tracks, densification, matching and band constraints', async () => {
  const props: TrackSimilarityOperationProps = {
    linesA: {positions: 'positionsA', lineOffsets: 'offsetsA'},
    linesB: {positions: 'positionsB', lineOffsets: 'offsetsB'},
    pairA: 'pairA',
    pairB: 'pairB',
    pairCount: 3,
    frechetEpsilon: 'epsilon',
    densify: 0.5,
    dynamicTimeWarpingBandRadius: 0,
    lcssBandRadius: 1,
    lcssTolerance: 1.1,
    editDistanceTolerance: 0.75,
    editDistanceGapPoint: [1, -1],
    output: OUTPUT
  };
  const rows = await runComparison(
    {
      positionsA: {format: 'float32x2', values: Float32Array.of(0, 0, 2, 2, 4, 0, 1, 1, 3, 0)},
      offsetsA: {format: 'uint32', values: Uint32Array.of(0, 3, 5)},
      positionsB: {
        format: 'float32x2',
        values: Float32Array.of(0, 1, 1, 0, 3, 0, 4, 1, 1, 2, 3, 2, 5, 1)
      },
      offsetsB: {format: 'uint32', values: Uint32Array.of(0, 4, 7)},
      pairA: {format: 'uint32', values: Uint32Array.of(0, 1, 0)},
      pairB: {format: 'uint32', values: Uint32Array.of(0, 1, 1)},
      epsilon: {format: 'float32', values: Float32Array.of(2.1)}
    },
    props
  );
  expect(rows['warping'][0]).toBeGreaterThan(rows['frechet'][0]);
  expect(rows['edits'][0]).toBeGreaterThan(0);
  expect(rows['subsequence'][0]).toBeGreaterThan(0);
  expect(rows['subsequence'][0]).toBeLessThan(1);
});

it.each([
  {epsilon: 0, expectedStatus: [0, 1, 4, 4], expectedWithin: [1, 0, 0, 0]},
  {epsilon: -1, expectedStatus: [16, 17, 4, 4], expectedWithin: [0, 0, 0, 0]}
])('compares empty tracks, out-of-range pairs and inactive capacity rows (epsilon $epsilon)', async ({
  epsilon,
  expectedStatus,
  expectedWithin
}) => {
  // The identical pair has distance zero. Mask 1 marks the empty track. Mask 4 alone marks
  // an out-of-range or inactive pair. Negative epsilon adds mask 16 to the first two rows.
  const rows = await runComparison(
    {
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, 2, 0)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 0, 2)},
      pairA: {format: 'uint32', values: Uint32Array.of(1, 0, 9, 1)},
      pairB: {format: 'uint32', values: Uint32Array.of(1, 1, 1, 1)},
      active: {format: 'uint32', values: Uint32Array.of(3)},
      epsilon: {format: 'float32', values: Float32Array.of(epsilon)}
    },
    {
      linesA: {positions: 'positions', lineOffsets: 'offsets'},
      pairA: 'pairA',
      pairB: 'pairB',
      activePairCount: 'active',
      pairCount: 4,
      frechetEpsilon: 'epsilon',
      output: OUTPUT
    }
  );
  expect(Array.from(rows['status'])).toEqual(expectedStatus);
  expect(Array.from(rows['within'])).toEqual(expectedWithin);
  expect(Array.from(rows['frechet'])).toEqual([0, NaN, NaN, NaN]);
});

it('both track similarity forms honor an explicit one-vertex dynamic-program cap', async () => {
  const rows = await runComparison(
    {
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
      pairA: {format: 'uint32', values: Uint32Array.of(0)},
      pairB: {format: 'uint32', values: Uint32Array.of(0)},
      epsilon: {format: 'float32', values: Float32Array.of(1)}
    },
    {
      linesA: {positions: 'positions', lineOffsets: 'offsets'},
      pairA: 'pairA',
      pairB: 'pairB',
      pairCount: 1,
      maximumFrechetVertexCount: 1,
      frechetEpsilon: 'epsilon',
      output: OUTPUT
    }
  );
  expect(Array.from(rows['status'])).toEqual([2]);
  expect(rows['frechet'][0]).toBeNaN();
  expect(rows['hausdorff'][0]).toBe(0);
  expect(rows['maximum'][0]).toBe(1);
});

it('bounds antimeridian segment measures by endpoints and retains the interior minimum', async () => {
  const rows = await runComparison(
    {
      positionsA: {format: 'float32x2', values: Float32Array.of(0, 0, -60, 0, 180, 0)},
      offsetsA: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3)},
      positionsB: {format: 'float32x2', values: Float32Array.of(100, 0, -100, 0, -100, 0, 100, 0)},
      offsetsB: {format: 'uint32', values: Uint32Array.of(0, 2, 4)},
      pairA: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 2)},
      pairB: {format: 'uint32', values: Uint32Array.of(0, 1, 0, 0)},
      epsilon: {format: 'float32', values: Float32Array.of(101)}
    },
    {
      linesA: {positions: 'positionsA', lineOffsets: 'offsetsA'},
      linesB: {positions: 'positionsB', lineOffsets: 'offsetsB'},
      pairA: 'pairA',
      pairB: 'pairB',
      pairCount: 4,
      frechetEpsilon: 'epsilon',
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: 180 / Math.PI
      },
      output: OUTPUT
    }
  );
  // Point-to-segment minima are 100, 100, 40, 0 degrees. Reverse endpoint means are
  // 100, 100, (160 + 40) / 2, 80. Hausdorff takes the larger directed distance.
  expect(Array.from(rows['status'])).toEqual([0, 0, 0, 0]);
  expect(Array.from(rows['hausdorff'])).toEqual([100, 100, 160, 80]);
  expect(Array.from(rows['segmentPath'])).toEqual([100, 100, 70, 40]);
  expect(Array.from(rows['maximum'])).toEqual([100, 100, 160, 80]);
});

it('compares a multi-strip dynamic program and geographic seam/degenerate segment cases', async () => {
  const first = Float32Array.from({length: 300 * 2}, (_, index) =>
    index % 2 === 0 ? Math.floor(index / 2) / 8 : 0
  );
  await runComparison(
    {
      positionsA: {format: 'float32x2', values: first},
      offsetsA: {format: 'uint32', values: Uint32Array.of(0, 300)},
      positionsB: {format: 'float32x2', values: Float32Array.of(0, 1, 18, 1, 38, 1)},
      offsetsB: {format: 'uint32', values: Uint32Array.of(0, 3)},
      pairA: {format: 'uint32', values: Uint32Array.of(0)},
      pairB: {format: 'uint32', values: Uint32Array.of(0)},
      epsilon: {format: 'float32', values: Float32Array.of(2)}
    },
    {
      linesA: {positions: 'positionsA', lineOffsets: 'offsetsA'},
      linesB: {positions: 'positionsB', lineOffsets: 'offsetsB'},
      pairA: 'pairA',
      pairB: 'pairB',
      pairCount: 1,
      frechetEpsilon: 'epsilon',
      output: OUTPUT
    }
  );
  await runComparison(
    {
      positions: {format: 'float32x2', values: Float32Array.of(179, 0, -179, 0, 180, 0, -180, 0)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 4)},
      pairA: {format: 'uint32', values: Uint32Array.of(0, 1)},
      pairB: {format: 'uint32', values: Uint32Array.of(1, 1)},
      epsilon: {format: 'float32', values: Float32Array.of(20)}
    },
    {
      linesA: {positions: 'positions', lineOffsets: 'offsets'},
      pairA: 'pairA',
      pairB: 'pairB',
      pairCount: 2,
      frechetEpsilon: 'epsilon',
      densify: 0.5,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: 1000
      },
      output: OUTPUT
    }
  );
});
