// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import advancedFixture from '../../oracles/fixtures/network-movement-advanced.json';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';
import {
  GPUNetworkTimeRouting,
  prepareGPUNetworkRestrictionStates
} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';

it('compacts survivors stably, reports overflow, and ranks exact threshold matches with pair provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [0, 2, 4]) {
    const harness = createOptimizationHarness(device);
    const output = {
      distance: harness.output('float32', 4),
      lowerBound: harness.output('float32', 4),
      status: harness.output('uint32', 4),
      matched: harness.output('uint32', 4),
      survivorSummary: harness.output('uint32', 3),
      topKIndices: harness.output('uint32', 2),
      topKSummary: harness.output('uint32x2', 1)
    };
    const result = await harness.run(
      new GPUTrackSimilaritySearch({
        metric: 'frechet',
        survivorCapacity: capacity,
        topK: 2,
        linesA: {
          kind: 'lines',
          positions: harness.input('float32x2', Float32Array.of(0, 0, 1, 0)),
          lineOffsets: harness.input('uint32', Uint32Array.of(0, 2))
        },
        linesB: {
          kind: 'lines',
          positions: harness.input(
            'float32x2',
            Float32Array.of(10, 0, 11, 0, 0, 1, 1, 1, 0, 0.5, 1, 0.5, 0, 0.25, 1, 0.25)
          ),
          lineOffsets: harness.input('uint32', Uint32Array.of(0, 2, 4, 6, 8))
        },
        pairA: harness.input('uint32', Uint32Array.of(0, 0, 0, 0)),
        pairB: harness.input('uint32', Uint32Array.of(0, 1, 2, 3)),
        threshold: harness.input('float32', Float32Array.of(2)),
        output
      })
    );
    expect(result(output.survivorSummary)).toEqual([
      3,
      Math.min(capacity, 3),
      Math.max(0, 3 - capacity)
    ]);
    expect(result(output.status)).toEqual(
      capacity === 0 ? [1, 8, 8, 8] : capacity === 2 ? [1, 0, 0, 8] : [1, 0, 0, 0]
    );
    expect(result(output.topKIndices)).toEqual(
      capacity === 0 ? [0xffffffff, 0xffffffff] : capacity === 2 ? [2, 1] : [3, 2]
    );
    expect(result(output.topKSummary)).toEqual([Math.min(capacity, 2), capacity >= 3 ? 1 : 0]);
  }
});

it('piecewise FIFO routing waits at the first sample, interpolates arrival, and rejects non-FIFO schedules', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const costs = harness.input('float32x2', Float32Array.of(2, 4, 10, 10, 0, 2, 10, 12));
  const output = {
    arrivals: harness.output('float32', 3),
    predecessorTransitions: harness.output('uint32', 3),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkTimeRouting({
      costModel: 'piecewise-arrival',
      samplesPerTransition: 2,
      waitingPolicy: 'wait-until-first-sample',
      stateVertices: harness.input('uint32', Uint32Array.of(0, 1, 2)),
      transitions: harness.input('uint32x2', Uint32Array.of(0, 1, 1, 2)),
      costs,
      endpoints: harness.input('uint32', Uint32Array.of(0, 2)),
      departure: harness.input('float32', Float32Array.of(0)),
      output
    }),
    first => {
      expect(first(output.arrivals)).toEqual([0, 4, 6]);
      expect(first(output.summary).slice(0, 3)).toEqual([1, 1, 2]);
      harness.rewrite(costs, Float32Array.of(2, 12, 10, 10, 0, 2, 10, 12));
    }
  );
  expect(result(output.summary)[0]).toBe(0);
});

it('expanded restrictions preserve legal parallel histories across overlapping prohibited sequences', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const prepared = prepareGPUNetworkRestrictionStates(
    4,
    [
      {source: 0, target: 1},
      {source: 1, target: 2},
      {source: 2, target: 3},
      {source: 0, target: 1},
      {source: 1, target: 3}
    ],
    [
      [0, 1, 2],
      [3, 4]
    ]
  );
  const harness = createOptimizationHarness(device);
  const output = {
    arrivals: harness.output('float32', prepared.stateVertices.length),
    predecessorTransitions: harness.output('uint32', prepared.stateVertices.length),
    summary: harness.output('uint32', 4)
  };
  const baseCosts = [1, 1, 1, 2, 5];
  const result = await harness.run(
    new GPUNetworkTimeRouting({
      stateVertices: harness.input('uint32', prepared.stateVertices),
      transitions: harness.input('uint32x2', prepared.transitions),
      costs: harness.input(
        'float32x2',
        Float32Array.from(Array.from(prepared.originalEdges).flatMap(edge => [baseCosts[edge], 0]))
      ),
      endpoints: harness.input('uint32', Uint32Array.of(prepared.initialStates[0], 3)),
      departure: harness.input('float32', Float32Array.of(0)),
      output
    })
  );
  expect(new Float32Array(Uint32Array.of(result(output.summary)[3]).buffer)[0]).toBe(4);
});

it('directed seeded Louvain handles incoming-only neighborhoods and asymmetric self loops', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    communities: harness.output('uint32', 4),
    objectiveTrace: harness.output('float32', 129),
    summary: harness.output('uint32', 4)
  };
  const weights = [2, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 1];
  const result = await harness.run(
    new GPUNetworkLouvain({
      nodeCount: 4,
      directed: true,
      seed: 17,
      weights: harness.input('float32', Float32Array.from(weights)),
      parameters: harness.input('float32', Float32Array.of(1, 0.000001)),
      output
    })
  );
  const labels = result(output.communities);
  expect(labels[0]).toBe(labels[1]);
  expect(labels[2]).toBe(labels[3]);
  expect(labels[0]).not.toBe(labels[2]);
  const trace = result(output.objectiveTrace).filter(Number.isFinite);
  expect(trace.at(-1)).toBeCloseTo(1 - (8 / 14) ** 2 - (6 / 14) ** 2, 5);
  for (let index = 1; index < trace.length; index++)
    expect(trace[index]).toBeGreaterThanOrEqual(trace[index - 1] - 1e-6);
});

it('polygon holes produce separated open/closed intervals, compact provenance, and explicit capacity loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const slots of [1, 8]) {
    const harness = createOptimizationHarness(device);
    const output = {
      intervals: harness.output('float32x2', 2 * slots),
      positions: harness.output('float32x4', 2 * slots),
      status: harness.output('uint32', 2 * slots),
      relationships: harness.output('uint32x4', 1),
      compactIntervals: harness.output('float32x2', 8),
      compactPositions: harness.output('float32x4', 8),
      compactSources: harness.output('uint32x2', 8),
      compactStatus: harness.output('uint32', 8),
      compactSummary: harness.output('uint32', 3)
    };
    const result = await harness.run(
      new GPUTrajectoryRestriction({
        positions: harness.input('float32x2', Float32Array.of(-1, 2, 5, 2)),
        times: harness.input('float32', Float32Array.of(0, 6)),
        offsets: harness.input('uint32', Uint32Array.of(0, 2)),
        parameters: harness.input(
          'float32',
          Float32Array.of(0, 6, 0, 0, 4, 4, 10, 0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 3, 1, 3, 3, 1, 3)
        ),
        polygonRingVertexCounts: [4, 4],
        intervalsPerSegment: slots,
        geometryBoundary: 'open',
        output
      })
    );
    expect(result(output.relationships).slice(0, 2)).toEqual([1, 0]);
    expect(result(output.intervals).slice(0, 2)).toEqual([1, 2]);
    expect(result(output.status)[0] & 192).toBe(192);
    if (slots === 8) {
      expect(result(output.compactIntervals).slice(0, 4)).toEqual(
        advancedFixture.clipIntervals.flat()
      );
      expect(result(output.compactSummary)).toEqual([2, 2, 0]);
    } else {
      expect(result(output.status)[0] & 256).toBe(256);
      expect(result(output.compactSummary)[2]).toBe(1);
    }
  }
});

it('full correlated Kalman covariance and RTS match independent NumPy matrix solves', async () => {
  const {GPUTrajectoryKalman} = await import(
    '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman'
  );
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-advanced.json'
  );
  const fixture = reference.fullCovariance;
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    filteredState: harness.output('float32x4', 4),
    filteredCovariance: harness.output('float32x4', 16),
    smoothedState: harness.output('float32x4', 4),
    smoothedCovariance: harness.output('float32x4', 16),
    status: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUTrajectoryKalman({
      covarianceMode: 'full',
      positions: harness.input('float32x2', Float32Array.from(fixture.positions.flat())),
      times: harness.input('float32', Float32Array.from(fixture.times)),
      offsets: harness.input('uint32', Uint32Array.of(0, 4)),
      parameters: harness.input('float32', Float32Array.from(fixture.parameters)),
      output
    })
  );
  for (const key of [
    'filteredState',
    'filteredCovariance',
    'smoothedState',
    'smoothedCovariance'
  ] as const)
    result(output[key]).forEach((value, index) =>
      expect(value).toBeCloseTo(fixture[key][index], 4)
    );
  expect(result(output.status)).toEqual([1, 0, 0, 0]);
});

it('Kalman smoothing matches direct MovingPandas 0.22.4 with its zero velocity prior', async () => {
  const {GPUTrajectoryKalman} = await import(
    '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman'
  );
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-advanced.json'
  );
  const fixture = reference.movingpandasKalman;
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    filteredState: harness.output('float32x4', 4),
    filteredCovariance: harness.output('float32x4', 4),
    smoothedState: harness.output('float32x4', 4),
    smoothedCovariance: harness.output('float32x4', 4),
    status: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUTrajectoryKalman({
      positions: harness.input('float32x2', Float32Array.from(fixture.positions.flat())),
      times: harness.input('float32', Float32Array.from(fixture.times)),
      offsets: harness.input('uint32', Uint32Array.of(0, 4)),
      parameters: harness.input('float32', Float32Array.from(fixture.parameters)),
      output
    })
  );
  fixture.smoothedPositions.forEach((point, row) =>
    point.forEach((value, axis) =>
      expect(result(output.smoothedState)[row * 4 + axis]).toBeCloseTo(value, 4)
    )
  );
});

it('trajectory search matches direct tslearn Euclidean-cost DTW/LCSS and similaritymeasures Frechet', async () => {
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-advanced.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const metric of ['dynamic-time-warping', 'frechet', 'longest-common-subsequence'] as const) {
    const harness = createOptimizationHarness(device);
    const count = reference.similarity.length;
    const values = harness.output('float32', count);
    const bounds = harness.output('float32', count);
    const output = {
      distance: metric !== 'longest-common-subsequence' ? values : undefined,
      lowerBound: metric !== 'longest-common-subsequence' ? bounds : undefined,
      similarity: metric === 'longest-common-subsequence' ? values : undefined,
      upperBound: metric === 'longest-common-subsequence' ? bounds : undefined,
      matched: harness.output('uint32', count),
      status: harness.output('uint32', count)
    };
    const offsets = [0];
    for (const track of reference.tracks) offsets.push(offsets.at(-1)! + track.length);
    const result = await harness.run(
      new GPUTrackSimilaritySearch({
        metric,
        lcssTolerance: 0.5,
        linesA: {
          kind: 'lines',
          positions: harness.input('float32x2', Float32Array.from(reference.tracks.flat(2))),
          lineOffsets: harness.input('uint32', Uint32Array.from(offsets))
        },
        pairA: harness.input(
          'uint32',
          Uint32Array.from(reference.similarity.map(row => row.pair[0]))
        ),
        pairB: harness.input(
          'uint32',
          Uint32Array.from(reference.similarity.map(row => row.pair[1]))
        ),
        threshold: harness.input(
          'float32',
          Float32Array.of(metric === 'longest-common-subsequence' ? 0 : 100)
        ),
        output
      })
    );
    const key =
      metric === 'dynamic-time-warping' ? 'dtw' : metric === 'frechet' ? 'frechet' : 'lcss';
    result(values).forEach((value, row) =>
      expect(value).toBeCloseTo(reference.similarity[row][key], 5)
    );
    expect(result(output.status)).toEqual(Array(count).fill(0));
  }
});

it('RAAM scores agree with direct access 1.1.9 at an equilibrium', async () => {
  const {GPUNetworkRationalAccess} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-rational-access'
  );
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-advanced.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const fixture = reference.raam;
  const output = {
    assignments: harness.output('float32', 4),
    scores: harness.output('float32', 2),
    unservedDemand: harness.output('float32', 2),
    status: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkRationalAccess({
      costs: harness.input('float32', Float32Array.from(fixture.costs)),
      demand: harness.input('float32', Float32Array.from(fixture.demand)),
      supply: harness.input('float32', Float32Array.from(fixture.supply)),
      parameters: harness.input('float32', Float32Array.of(10, 1, 0.2, 0.005, 50, 0)),
      output
    })
  );
  result(output.scores).forEach((value, row) => expect(value).toBeCloseTo(fixture.scores[row], 5));
});

it('pairwise temporal distance returns analytic crossing intervals and unknown zero support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 16),
    positions: harness.output('float32x4', 16),
    status: harness.output('uint32', 16),
    relationships: harness.output('uint32x4', 2)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', Float32Array.of(-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)),
      times: harness.input('float32', Float32Array.of(0, 4, 0, 4, 10, 11)),
      offsets: harness.input('uint32', Uint32Array.of(0, 2, 4, 6)),
      parameters: harness.input('float32', Float32Array.of(0, 20, 1, 0, 0, 0, 5)),
      pairwiseTracks: [
        [0, 1],
        [0, 2]
      ],
      intervalsPerPair: 8,
      geometryBoundary: 'open',
      output
    })
  );
  expect(result(output.intervals).slice(0, 2)).toEqual([1, 3]);
  expect(result(output.status)[0] & 192).toBe(192);
  expect(result(output.relationships).slice(0, 2)).toEqual([1, 0]);
  expect(result(output.relationships).slice(4, 6)).toEqual([0xffffffff, 0xffffffff]);
});

it('exact flow objectives exceed f32 integer precision without rounding the certificate', async () => {
  const {GPUNetworkFlow} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-flow'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    flow: harness.output('uint32', 1),
    cut: harness.output('uint32', 2),
    summary: harness.output('uint32', 8),
    exactCost: harness.output('uint32x2', 1)
  };
  const result = await harness.run(
    new GPUNetworkFlow({
      nodeCount: 2,
      edges: harness.input('uint32x2', Uint32Array.of(0, 1)),
      capacities: harness.input('uint32', Uint32Array.of(0xffffff)),
      costs: harness.input('uint32', Uint32Array.of(32767)),
      parameters: harness.input('uint32', Uint32Array.of(0, 1, 0xffffffff)),
      output
    })
  );
  const exact = result(output.exactCost);
  expect(BigInt(exact[0]) + (BigInt(exact[1]) << 32n)).toBe(0xffffffn * 32767n);
  expect(result(output.summary)[7]).toBe(1);
});

it('fractional rational access balances nonintegral demand with an absolute shift cap', async () => {
  const {GPUNetworkRationalAccess} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-rational-access'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    assignments: harness.output('float32', 2),
    scores: harness.output('float32', 1),
    unservedDemand: harness.output('float32', 1),
    status: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkRationalAccess({
      demandMode: 'fractional',
      stepMode: 'absolute',
      initialLimitCycles: 0,
      costs: harness.input('float32', Float32Array.of(0, 0)),
      demand: harness.input('float32', Float32Array.of(2.5)),
      supply: harness.input('float32', Float32Array.of(1, 1)),
      parameters: harness.input('float32', Float32Array.of(1, 1, 2, 0.5, 50, 0.000001)),
      output
    })
  );
  expect(result(output.assignments)).toEqual([1.25, 1.25]);
  expect(result(output.status).slice(0, 2)).toEqual([1, 1]);
});

it('temporal interval union merges across segments, preserves track boundaries, and counts after capacity exhaustion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 6),
    positions: harness.output('float32x4', 6),
    status: harness.output('uint32', 6),
    relationships: harness.output('uint32x4', 2),
    compactIntervals: harness.output('float32x2', 1),
    compactPositions: harness.output('float32x4', 1),
    compactSources: harness.output('uint32x2', 1),
    compactStatus: harness.output('uint32', 1),
    compactSummary: harness.output('uint32', 3)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', Float32Array.of(0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 2, 0)),
      times: harness.input('float32', Float32Array.of(0, 1, 2, 0, 1, 2)),
      offsets: harness.input('uint32', Uint32Array.of(0, 3, 6)),
      parameters: harness.input('float32', Float32Array.of(0, 2, -1, -1, 3, 1, 5)),
      mergeIntervals: true,
      output
    })
  );
  expect(result(output.compactIntervals)).toEqual([0, 2]);
  expect(result(output.compactSources)).toEqual([0, 2]);
  expect(result(output.compactPositions)).toEqual([0, 0, 2, 0]);
  expect(result(output.compactSummary)).toEqual([2, 1, 1]);
});

it('re-encoding similarity search clears compacted survivors, values and top-k after threshold changes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const threshold = harness.input('float32', Float32Array.of(2));
  const output = {
    distance: harness.output('float32', 1),
    lowerBound: harness.output('float32', 1),
    matched: harness.output('uint32', 1),
    status: harness.output('uint32', 1),
    survivorSummary: harness.output('uint32', 3),
    topKIndices: harness.output('uint32', 2),
    topKSummary: harness.output('uint32x2', 2)
  };
  const result = await harness.run(
    new GPUTrackSimilaritySearch({
      metric: 'frechet',
      topK: 1,
      linesA: {
        kind: 'lines',
        positions: harness.input('float32x2', Float32Array.of(0, 0, 0, 1)),
        lineOffsets: harness.input('uint32', Uint32Array.of(0, 1, 2))
      },
      pairA: harness.input('uint32', Uint32Array.of(0)),
      pairB: harness.input('uint32', Uint32Array.of(1)),
      threshold,
      output
    }),
    first => {
      expect(first(output.distance)).toEqual([1]);
      expect(first(output.topKIndices)).toEqual([0, 0xffffffff]);
      harness.rewrite(threshold, Float32Array.of(0.5));
    }
  );
  expect(result(output.survivorSummary)).toEqual([0, 0, 0]);
  expect(result(output.distance)[0]).toBeNaN();
  expect(result(output.topKIndices)).toEqual([0xffffffff, 0xffffffff]);
  expect(result(output.status)).toEqual([1]);
});

it('routing matches committed independent NetworkX virtual-split and time-expanded references', async () => {
  const {GPUNetworkPointRouting} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-point-routing'
  );
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-advanced.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const point = createOptimizationHarness(device);
  const fixture = reference.pointRouting;
  const pointOutput = {
    costs: point.output('float32', 1),
    status: point.output('uint32', 1),
    predecessorEdges: point.output('uint32', 3)
  };
  const pointResult = await point.run(
    new GPUNetworkPointRouting({
      costs: point.input('float32', Float32Array.from(fixture.edgeCosts)),
      turns: point.input('uint32x2', Uint32Array.from(fixture.turns.flat())),
      queries: point.input('uint32x2', Uint32Array.from(fixture.queries.flat())),
      fractions: point.input('float32x2', Float32Array.from(fixture.fractions.flat())),
      output: pointOutput
    })
  );
  expect(pointResult(pointOutput.costs)).toEqual([fixture.cost]);
  const time = createOptimizationHarness(device);
  const timeOutput = {
    arrivals: time.output('float32', 3),
    predecessorTransitions: time.output('uint32', 3),
    summary: time.output('uint32', 4)
  };
  const timeResult = await time.run(
    new GPUNetworkTimeRouting({
      stateVertices: time.input('uint32', Uint32Array.of(0, 1, 2)),
      transitions: time.input('uint32x2', Uint32Array.of(0, 1, 1, 2)),
      costs: time.input('float32x2', Float32Array.of(2, 1, 1, 0)),
      endpoints: time.input('uint32', Uint32Array.of(0, 2)),
      departure: time.input('float32', Float32Array.of(reference.timeRouting.departure)),
      output: timeOutput
    })
  );
  expect(timeResult(timeOutput.arrivals)[2]).toBe(reference.timeRouting.arrival);
});

it('all three bounded facility objectives match direct spopt 0.7.0 CBC solutions', async () => {
  const {GPUNetworkFacilityOptimization} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization'
  );
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-advanced.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['backup-coverage', 'p-dispersion', 'capacitated-p-median'] as const) {
    const fixture = reference.facility.find(row => row.operation === operation)!;
    const harness = createOptimizationHarness(device);
    const output = {
      selectedFacilities: harness.output('uint32', fixture.facilityCount),
      assignments: harness.output('uint32', fixture.demand.length),
      objective: harness.output('float32', 1),
      status: harness.output('uint32', 4)
    };
    const result = await harness.run(
      new GPUNetworkFacilityOptimization({
        operation,
        facilityCount: fixture.facilityCount,
        selectCount: fixture.selectCount ?? undefined,
        costs: harness.input('float32', Float32Array.from(fixture.costs.flat())),
        demand: harness.input('float32', Float32Array.from(fixture.demand)),
        capacities: harness.input('float32', Float32Array.from(fixture.capacities)),
        threshold: harness.input('float32', Float32Array.of(fixture.threshold)),
        output
      })
    );
    expect(result(output.objective)).toEqual([fixture.objective]);
    expect(result(output.status).slice(0, 3)).toEqual([1, 1, 1]);
  }
});

it('full covariance scales the innovation inverse instead of silently accepting an overflowing determinant', async () => {
  const {GPUTrajectoryKalman} = await import(
    '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const covarianceMode of ['full', 'isotropic'] as const) {
    const harness = createOptimizationHarness(device);
    const output = {
      filteredState: harness.output('float32x4', 2),
      filteredCovariance: harness.output('float32x4', covarianceMode === 'full' ? 8 : 2),
      status: harness.output('uint32', 2)
    };
    const result = await harness.run(
      new GPUTrajectoryKalman({
        covarianceMode,
        positions: harness.input('float32x2', Float32Array.of(0, 0, 2, 0)),
        times: harness.input('float32', Float32Array.of(0, 1)),
        offsets: harness.input('uint32', Uint32Array.of(0, 2)),
        parameters: harness.input(
          'float32',
          covarianceMode === 'full'
            ? Float32Array.of(0, 0, 0, 1e30, 0, 1e30, 0, 0, 0, 10)
            : Float32Array.of(0, 3e38, 0, 10)
        ),
        output
      })
    );
    if (covarianceMode === 'full') {
      expect(result(output.filteredState)[4]).toBeCloseTo(1, 5);
      expect(result(output.status)).toEqual([1, 0]);
    } else {
      expect(result(output.status)).toEqual([1, 4]);
      expect(result(output.filteredState)[4]).toBeNaN();
    }
  }
});

it('pairwise intervals preserve narrow encounters on long map-scale segments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 8),
    positions: harness.output('float32x4', 8),
    status: harness.output('uint32', 8),
    relationships: harness.output('uint32x4', 1)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', Float32Array.of(100000, 0, -100000, 0, 0, 0, 0, 0)),
      times: harness.input('float32', Float32Array.of(0, 200000, 0, 200000)),
      offsets: harness.input('uint32', Uint32Array.of(0, 2, 4)),
      parameters: harness.input('float32', Float32Array.of(0, 200000, 1, 0, 0, 0, 300000)),
      pairwiseTracks: [[0, 1]],
      intervalsPerPair: 8,
      output
    })
  );
  expect(result(output.intervals).slice(0, 2)).toEqual([99999, 100001]);
  expect(result(output.status)[0] & 7).toBe(1);
});

it('fractional directed routes match direct pgRouting 3.8.0 withPointsCost', async () => {
  const {GPUNetworkPointRouting} = await import(
    '../../../src/gpu-network/network-optimization/gpu-network-point-routing'
  );
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-postgres.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    costs: harness.output('float32', 4),
    status: harness.output('uint32', 4),
    predecessorEdges: harness.output('uint32', 12)
  };
  const result = await harness.run(
    new GPUNetworkPointRouting({
      costs: harness.input('float32', Float32Array.of(4, 4, 6)),
      turns: harness.input('uint32x2', Uint32Array.of(0, 1, 2, 1)),
      queries: harness.input(
        'uint32x2',
        Uint32Array.from(reference.routing.flatMap(entry => entry.query))
      ),
      fractions: harness.input(
        'float32x2',
        Float32Array.from(reference.routing.flatMap(entry => entry.fractions))
      ),
      output
    })
  );
  expect(result(output.costs)).toEqual(reference.routing.map(entry => entry.cost ?? Infinity));
  expect(result(output.status)).toEqual([0, 0, 0, 0]);
});

it('polygon holes and all four time endpoint policies match direct MobilityDB 1.2.1', async () => {
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-postgres.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const policies = ['closed', 'open', 'left-closed', 'right-closed'] as const;
  for (let index = 0; index < 5; index++) {
    const harness = createOptimizationHarness(device);
    const polygon = index === 4;
    const output = {
      intervals: harness.output('float32x2', 8),
      positions: harness.output('float32x4', 8),
      status: harness.output('uint32', 8),
      relationships: harness.output('uint32x4', 1)
    };
    const result = await harness.run(
      new GPUTrajectoryRestriction({
        positions: harness.input('float32x2', Float32Array.of(-1, 2, 5, 2)),
        times: harness.input('float32', Float32Array.of(0, 6)),
        offsets: harness.input('uint32', Uint32Array.of(0, 2)),
        parameters: harness.input(
          'float32',
          polygon
            ? Float32Array.of(0, 6, 0, 0, 4, 4, 10, 0, 0, 4, 0, 4, 4, 0, 4, 1, 1, 3, 1, 3, 3, 1, 3)
            : Float32Array.of(1, 5, -10, -10, 10, 10, 10)
        ),
        polygonRingVertexCounts: polygon ? [4, 4] : undefined,
        timeBoundary: polygon ? 'closed' : policies[index],
        geometryBoundary: 'closed',
        intervalsPerSegment: 4,
        output
      })
    );
    const expected = polygon ? reference.clip : [reference.timeBoundaries[index]];
    for (let interval = 0; interval < expected.length; interval++) {
      expect(result(output.intervals).slice(2 * interval, 2 * interval + 2)).toEqual(
        expected[interval].interval
      );
      expect(result(output.status)[interval] & 192).toBe(
        (expected[interval].closed[0] ? 0 : 64) + (expected[interval].closed[1] ? 0 : 128)
      );
    }
  }
});

it('continuous encounter intervals and three-valued support match direct MobilityDB temporal predicates', async () => {
  const {default: reference} = await import(
    '../../oracles/fixtures/network-movement-postgres.json'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    intervals: harness.output('float32x2', 16),
    positions: harness.output('float32x4', 16),
    status: harness.output('uint32', 16),
    relationships: harness.output('uint32x4', 2)
  };
  const result = await harness.run(
    new GPUTrajectoryRestriction({
      positions: harness.input('float32x2', Float32Array.of(-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0)),
      times: harness.input('float32', Float32Array.of(0, 4, 0, 4, 10, 11)),
      offsets: harness.input('uint32', Uint32Array.of(0, 2, 4, 6)),
      parameters: harness.input('float32', Float32Array.of(0, 20, 1, 0, 0, 0, 5)),
      pairwiseTracks: [
        [0, 1],
        [0, 2]
      ],
      intervalsPerPair: 8,
      geometryBoundary: 'closed',
      output
    })
  );
  expect(result(output.intervals).slice(0, 2)).toEqual(reference.pairwise.interval);
  expect(result(output.status)[0] & 192).toBe(0);
  expect(result(output.relationships).slice(0, 2)).toEqual([
    Number(reference.pairwise.ever),
    Number(reference.pairwise.always)
  ]);
  expect(result(output.relationships).slice(4, 6)).toEqual([0xffffffff, 0xffffffff]);
  expect(reference.pairwise.disjointEver).toBeNull();
  expect(reference.pairwise.disjointAlways).toBeNull();
  // MobilityDB 1.2.1 aDwithin disagrees with its full temporal predicate on this crossing.
  expect(reference.pairwise.directADwithin).toBe(true);
});

it('piecewise FIFO extrapolation preserves finite arrivals near the f32 range limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const output = {
    arrivals: harness.output('float32', 2),
    predecessorTransitions: harness.output('uint32', 2),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkTimeRouting({
      costModel: 'piecewise-arrival',
      samplesPerTransition: 2,
      stateVertices: harness.input('uint32', Uint32Array.of(0, 1)),
      transitions: harness.input('uint32x2', Uint32Array.of(0, 1)),
      costs: harness.input('float32x2', Float32Array.of(0, 1e38, 1e38, 2e38)),
      endpoints: harness.input('uint32', Uint32Array.of(0, 1)),
      departure: harness.input('float32', Float32Array.of(2e38)),
      output
    })
  );
  expect(result(output.summary).slice(0, 3)).toEqual([1, 1, 1]);
  expect(result(output.arrivals)[1] / 1e38).toBeCloseTo(3, 5);
});
