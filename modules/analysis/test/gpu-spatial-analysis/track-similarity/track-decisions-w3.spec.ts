// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity';
import {createDynamicProgramHarness} from './track-similarity-dtw-frechet-harness';
import {runMeasures, runSearch} from './track-similarity-edit-measures-harness';

it('keeps extreme planar norms and decisions on direct, named and retained search pairs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const distance of [2 ** -80, 2 ** 70]) {
    // Powers of two and their halves are exact binary32 values; d > d/2.
    const direct = createDynamicProgramHarness(device, [[[0, 0]]], [[[distance, 0]]]);
    try {
      const result = await direct.run(distance / 2);
      expect(Array.from(result.frechet)).toEqual([distance]);
      expect(Array.from(result.within)).toEqual([0]);
      expect(Array.from(result.status)).toEqual([0]);
    } finally {
      direct.destroy();
    }
    const named = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, distance, 0)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
        first: {format: 'uint32', values: Uint32Array.of(0)},
        second: {format: 'uint32', values: Uint32Array.of(1)},
        epsilon: {format: 'float32', values: Float32Array.of(distance / 2)}
      },
      operations: [
        new GPUTrackSimilarityOperation({
          linesA: {positions: 'positions', lineOffsets: 'offsets'},
          pairA: 'first',
          pairB: 'second',
          pairCount: 1,
          frechetEpsilon: 'epsilon',
          output: {frechet: 'distance', frechetWithin: 'within', pairStatus: 'status'}
        })
      ],
      outputs: ['distance', 'within', 'status']
    });
    expect(Array.from(named['distance'])).toEqual([distance]);
    expect(Array.from(named['within'])).toEqual([0]);
    expect(Array.from(named['status'])).toEqual([0]);
    // Equal endpoints force the endpoint lower bound to retain this pair.
    const search = await runSearch(
      device,
      [
        [
          [0, 0],
          [distance, 0],
          [0, 0]
        ]
      ],
      [[[0, 0]]],
      Uint32Array.of(0),
      Uint32Array.of(0),
      {metric: 'frechet', threshold: distance / 2, topK: 1}
    );
    expect(Array.from(search.distance)).toEqual([distance]);
    expect(Array.from(search.matched)).toEqual([0]);
    expect(Array.from(search.status)).toEqual([0]);
  }
});

it('preserves the small component at radius one, and includes exact equality', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const height of [0, 2 ** -13]) {
    // Squared distance is exactly 1 + 2^-26 for the nonzero height.
    const direct = createDynamicProgramHarness(device, [[[0, 0]]], [[[1, height]]]);
    try {
      const result = await direct.run(1);
      expect(Array.from(result.frechet)).toEqual([1]);
      expect(Array.from(result.within)).toEqual([height ? 0 : 1]);
      expect(Array.from(result.status)).toEqual([0]);
    } finally {
      direct.destroy();
    }
    const edits = await runMeasures(
      device,
      [[[0, 0]]],
      [[[1, height]]],
      ['editDistanceOnRealSequence', 'longestCommonSubsequence']
    );
    expect(Array.from(edits.editDistanceOnRealSequence)).toEqual([height ? 1 : 0]);
    expect(Array.from(edits.longestCommonSubsequence)).toEqual([height ? 0 : 1]);
    expect(Array.from(edits.status)).toEqual([0]);
  }
});

it('reports resident invalid geometry, threshold and cap causes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [longitude, latitude, flags] of [
    [NaN, NaN, 8],
    [5, 91, 8],
    [5, 90, 0]
  ]) {
    const harness = createDynamicProgramHarness(
      device,
      [[[longitude, latitude]]],
      [[[longitude, latitude]]],
      {spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}}
    );
    try {
      const result = await harness.run(1);
      expect(Array.from(result.status)).toEqual([flags]);
      expect(Array.from(result.within)).toEqual([flags ? 0 : 1]);
      if (!flags) expect(Array.from(result.frechet)).toEqual([0]);
    } finally {
      harness.destroy();
    }
  }
  for (const cap of [1, 2]) {
    const harness = createDynamicProgramHarness(
      device,
      [
        [
          [0, 0],
          [1, 0]
        ]
      ],
      [
        [
          [0, 0],
          [1, 0]
        ]
      ],
      {maximumFrechetVertexCount: cap}
    );
    try {
      for (const epsilon of [1, NaN]) {
        const result = await harness.run(epsilon);
        expect(Array.from(result.status)).toEqual([
          (cap === 1 ? 2 : 0) | (Number.isNaN(epsilon) ? 16 : 0)
        ]);
        expect(Array.from(result.within)).toEqual([cap === 2 && Number.isFinite(epsilon) ? 1 : 0]);
        if (cap === 2) expect(Array.from(result.frechet)).toEqual([0]);
      }
    } finally {
      harness.destroy();
    }
  }
});

it('keeps uncapped Hausdorff usable and singleton metric outputs current', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const capped = await runMeasures(
    device,
    [
      [
        [0, 0],
        [1, 0]
      ]
    ],
    [
      [
        [0, 0],
        [1, 0]
      ]
    ],
    ['hausdorff', 'editDistanceOnRealSequence'],
    {maximumFrechetVertexCount: 1}
  );
  expect(Array.from(capped.hausdorff)).toEqual([0]);
  expect(Array.from(capped.status)).toEqual([2]);
  expect(Array.from(capped.editDistanceOnRealSequence)).toEqual([NaN]);
  const singleton = createDynamicProgramHarness(device, [[[0, 0]]], [[[0, 0]]]);
  try {
    expect((await singleton.run(0)).frechet[0]).toBe(0);
  } finally {
    singleton.destroy();
  }
  for (const metric of ['frechet', 'longest-common-subsequence'] as const) {
    const search = await runSearch(
      device,
      [[[0, 0]]],
      [[[0, 0]]],
      Uint32Array.of(0),
      Uint32Array.of(0),
      {metric, threshold: metric === 'frechet' ? 0 : 1, topK: 1}
    );
    expect(Array.from(search.distance)).toEqual([metric === 'frechet' ? 0 : 1]);
    expect(Array.from(search.matched)).toEqual([1]);
    expect(Array.from(search.status)).toEqual([0]);
  }
  const matches = await runMeasures(device, [[[0, 0]]], [[[0, 0]]], ['longestCommonSubsequence']);
  expect(Array.from(matches.longestCommonSubsequence)).toEqual([1]);
  expect(Array.from(matches.status)).toEqual([0]);
});

it('preserves zero pair cardinality and distinguishes empty from absent tracks in the named GPU form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const offsets of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
    for (const pairs of [new Uint32Array(), Uint32Array.of(0)]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array()},
          offsets: {format: 'uint32', values: offsets},
          pairs: {format: 'uint32', values: pairs}
        },
        operations: [
          new GPUTrackSimilarityOperation({
            linesA: {positions: 'positions', lineOffsets: 'offsets'},
            pairA: 'pairs',
            pairB: 'pairs',
            pairCount: pairs.length,
            output: {frechet: 'distance', pairStatus: 'status'}
          })
        ],
        outputs: ['distance', 'status']
      });
      expect(result['distance'].length).toBe(pairs.length);
      expect(Array.from(result['status'])).toEqual(
        pairs.length ? [offsets.length === 1 ? 4 : 1] : []
      );
    }
  }
});

it('review: keeps non-axial equality and rejects sub-norm components in all decisions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [horizontal, vertical, radius, expected] of [
    [3, 4, 5, 1],
    [1, 2 ** -30, 1, 0]
  ]) {
    const harness = createDynamicProgramHarness(device, [[[0, 0]]], [[[horizontal, vertical]]]);
    try {
      const result = await harness.run(radius);
      expect(Array.from(result.within)).toEqual([expected]);
      expect(Array.from(result.frechet)).toEqual([radius]);
      expect(Array.from(result.status)).toEqual([0]);
    } finally {
      harness.destroy();
    }
    const measures = await runMeasures(
      device,
      [[[0, 0]]],
      [[[horizontal, vertical]]],
      ['editDistanceOnRealSequence', 'longestCommonSubsequence'],
      {editDistanceTolerance: radius, lcssTolerance: radius}
    );
    expect(Array.from(measures.editDistanceOnRealSequence)).toEqual([1 - expected]);
    expect(Array.from(measures.longestCommonSubsequence)).toEqual([expected]);
  }
});

it('review: keeps exact LCSS and extreme endpoint matches and rejects retained rounded matches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pairs = Uint32Array.of(0);
  const lcss = await runSearch(
    device,
    [[[0, 0]]],
    [[[20055 / 16384, 12728 / 16384]]],
    pairs,
    pairs,
    {metric: 'longest-common-subsequence', lcssTolerance: 23753 / 16384, threshold: 1, topK: 1}
  );
  expect(Array.from(lcss.distance)).toEqual([1]);
  expect(Array.from(lcss.matched)).toEqual([1]);
  expect(Array.from(lcss.status)).toEqual([0]);
  const endpoint = await runSearch(device, [[[0, 0]]], [[[2 ** 70, 0]]], pairs, pairs, {
    metric: 'frechet',
    threshold: 2 ** 70,
    topK: 1
  });
  expect(Array.from(endpoint.distance)).toEqual([2 ** 70]);
  expect(Array.from(endpoint.matched)).toEqual([1]);
  expect(Array.from(endpoint.status)).toEqual([0]);
  const retained = await runSearch(
    device,
    [
      [
        [0, 0],
        [1, 2 ** -30],
        [0, 0]
      ]
    ],
    [[[0, 0]]],
    pairs,
    pairs,
    {metric: 'frechet', threshold: 1, topK: 1}
  );
  expect(Array.from(retained.distance)).toEqual([1]);
  expect(Array.from(retained.matched)).toEqual([0]);
  expect(Array.from(retained.status)).toEqual([0]);
  const invalid = await runSearch(device, [[[0, 0]]], [[[0, 91]]], pairs, pairs, {
    metric: 'frechet',
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    threshold: 1,
    topK: 1
  });
  expect(Array.from(invalid.status)).toEqual([2]);
  expect(invalid.topKSummary[1]).toBe(0);
});

it('review: computes a fallback coupling and tiny self-line Hausdorff', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createDynamicProgramHarness(
    device,
    [
      [
        [0, 0],
        [0, 0],
        [1, 0]
      ]
    ],
    [
      [
        [0, 0],
        [1, 0],
        [1, 0]
      ]
    ]
  );
  try {
    expect(Array.from((await harness.run(0)).within)).toEqual([1]);
  } finally {
    harness.destroy();
  }
  const tiny = await runMeasures(
    device,
    [
      [
        [0, 0],
        [2 ** -80, 0]
      ]
    ],
    [
      [
        [0, 0],
        [2 ** -80, 0]
      ]
    ],
    ['hausdorff']
  );
  expect(Array.from(tiny.hausdorff)).toEqual([0]);
  expect(Array.from(tiny.status)).toEqual([0]);
});

it('review: pins cause precedence and multipart empty and absent tracks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const causes = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, NaN, 0)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
      first: {format: 'uint32', values: Uint32Array.of(0, 1, 0)},
      second: {format: 'uint32', values: Uint32Array.of(0, 0, 0)},
      active: {format: 'uint32', values: Uint32Array.of(2)},
      epsilon: {format: 'float32', values: Float32Array.of(NaN)}
    },
    operations: [
      new GPUTrackSimilarityOperation({
        linesA: {positions: 'positions', lineOffsets: 'offsets'},
        pairA: 'first',
        pairB: 'second',
        pairCount: 3,
        activePairCount: 'active',
        frechetEpsilon: 'epsilon',
        maximumFrechetVertexCount: 1,
        output: {frechet: 'distance', frechetWithin: 'within', pairStatus: 'status'}
      })
    ],
    outputs: ['distance', 'within', 'status']
  });
  expect(Array.from(causes['status'])).toEqual([24, 4, 4]);
  expect(Array.from(causes['distance'])).toEqual([NaN, NaN, NaN]);
  expect(Array.from(causes['within'])).toEqual([0, 0, 0]);
  for (const features of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array()},
        offsets: {format: 'uint32', values: Uint32Array.of(0)},
        features: {format: 'uint32', values: features},
        pairs: {format: 'uint32', values: Uint32Array.of(0)}
      },
      operations: [
        new GPUTrackSimilarityOperation({
          linesA: {positions: 'positions', lineOffsets: 'offsets', featureOffsets: 'features'},
          pairA: 'pairs',
          pairB: 'pairs',
          pairCount: 1,
          output: {hausdorff: 'distance', pairStatus: 'status'}
        })
      ],
      outputs: ['distance', 'status']
    });
    expect(Array.from(result['status'])).toEqual([features.length === 1 ? 4 : 1]);
    expect(Array.from(result['distance'])).toEqual([NaN]);
  }
});
