// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUFlowAggregation,
  GPU_FLOW_AGGREGATION_NO_ZONE,
  type GPUFlowAggregationProps
} from '../../../src/gpu-network/flow-aggregation/index';
import {getGPUPointDensityHexagonGridSize} from '../../../src/gpu-spatial-analysis/point-density/point-density-hexagon';
import {
  getGPUTimeWindowFilterParameterValues,
  getGPUTimeWindowFilterWordParameterValues,
  getInt64TimeWords,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH
} from '../../../src/dataframe-analysis/time-window-filter/index';
import {
  createInputBuffer,
  createVectorView,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  findNearestHexagon,
  createSeededPoints
} from '../../gpu-spatial-analysis/point-density/point-density-oracle';
import {
  computeFlowAggregation,
  createSeededRandom,
  type FlowOracleResult
} from './flow-aggregation-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const NO_ZONE = GPU_FLOW_AGGREGATION_NO_ZONE;
const GARBAGE = 0xdeadbeef;

type FlowResult = {
  ids: number[];
  count: number;
  requiredCount: number;
  overflow: number;
  pairOverflow: number;
  drawInstanceCount: number;
  originZones: number[];
  destinationZones: number[];
  counts: number[];
  weights: number[];
  zoneOutCounts: number[];
  zoneInCounts: number[];
  zoneOutWeights: number[];
  zoneInWeights: number[];
};

type Harness = {
  compiled: CompiledGPUCommandGraph<void>;
  submit(): void;
  read(): Promise<FlowResult>;
  destroy(): void;
};

type HarnessOptions = {
  id: string;
  zoneCount: number;
  topCount: number;
  pairCapacity: number;
  hasWeights: boolean;
  maximumProbeCount?: number;
  createProps: (
    graph: GPUCommandGraph<void>
  ) => Pick<GPUFlowAggregationProps, 'zones'> & Partial<GPUFlowAggregationProps>;
};

/** Builds one flow-aggregation graph with every output imported, scrambled before each submit. */
function createHarness(device: Device, options: HarnessOptions): Harness {
  const {topCount, zoneCount, hasWeights} = options;
  const buffers: Buffer[] = [];
  const outputs: {buffer: Buffer; length: number}[] = [];
  const graph = new GPUCommandGraph<void>(device, {id: options.id});
  const createOutput = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.push({buffer, length: Math.max(length, 1)});
    return {
      buffer,
      view: importGraphBuffer(graph, `${options.id}-${name}`, buffer, format, length)
    };
  };
  const ids = createOutput('ids', 'uint32', topCount);
  const count = createOutput('count', 'uint32', 1);
  const overflow = createOutput('overflow', 'uint32', 1);
  const requiredCount = createOutput('total', 'uint32', 1);
  const pairOverflow = createOutput('pair-overflow', 'uint32', 1);
  const drawInstanceCount = createOutput('draw', 'uint32', 1);
  const originZones = createOutput('origin-zones', 'uint32', topCount);
  const destinationZones = createOutput('destination-zones', 'uint32', topCount);
  const counts = createOutput('counts', 'uint32', topCount);
  const weights = createOutput('weights', 'float32', topCount);
  const zoneOutCounts = createOutput('zone-out-counts', 'uint32', zoneCount);
  const zoneInCounts = createOutput('zone-in-counts', 'uint32', zoneCount);
  const zoneOutWeights = createOutput('zone-out-weights', 'float32', zoneCount);
  const zoneInWeights = createOutput('zone-in-weights', 'float32', zoneCount);

  const contributor = new GPUFlowAggregation({
    id: options.id,
    pairCapacity: options.pairCapacity,
    maximumProbeCount: options.maximumProbeCount,
    ...options.createProps(graph),
    output: {
      ids: ids.view,
      status: {
        count: count.view,
        overflow: overflow.view,
        requiredCount: requiredCount.view,
        pairOverflow: pairOverflow.view
      },
      drawInstanceCount: drawInstanceCount.view,
      flowOriginZoneIds: originZones.view,
      flowDestinationZoneIds: destinationZones.view,
      flowCounts: counts.view,
      flowWeights: weights.view,
      zoneOutCounts: zoneOutCounts.view,
      zoneInCounts: zoneInCounts.view,
      ...(hasWeights
        ? {
            zoneOutWeights: zoneOutWeights.view,
            zoneInWeights: zoneInWeights.view
          }
        : {})
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    compiled,
    submit: () => {
      for (const output of outputs) {
        output.buffer.write(new Uint32Array(output.length).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
    },
    read: async () => ({
      ids: await readUint32(ids.buffer, topCount),
      count: (await readUint32(count.buffer, 1))[0],
      requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
      overflow: (await readUint32(overflow.buffer, 1))[0],
      pairOverflow: (await readUint32(pairOverflow.buffer, 1))[0],
      drawInstanceCount: (await readUint32(drawInstanceCount.buffer, 1))[0],
      originZones: await readUint32(originZones.buffer, topCount),
      destinationZones: await readUint32(destinationZones.buffer, topCount),
      counts: await readUint32(counts.buffer, topCount),
      weights: await readFloat32(weights.buffer, topCount),
      zoneOutCounts: await readUint32(zoneOutCounts.buffer, zoneCount),
      zoneInCounts: await readUint32(zoneInCounts.buffer, zoneCount),
      zoneOutWeights: await readFloat32(zoneOutWeights.buffer, zoneCount),
      zoneInWeights: await readFloat32(zoneInWeights.buffer, zoneCount)
    }),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Compares every output against the oracle for a table that did not overflow. */
function expectMatchesOracle(
  result: FlowResult,
  oracle: FlowOracleResult,
  topCount: number,
  hasWeights: boolean
): void {
  const distinct = oracle.flows.length;
  const count = Math.min(distinct, topCount);
  expect(result.count).toBe(count);
  expect(result.drawInstanceCount).toBe(count);
  expect(result.requiredCount).toBe(distinct);
  expect(result.pairOverflow).toBe(0);
  expect(result.overflow).toBe(distinct > topCount ? 1 : 0);
  for (let rank = 0; rank < topCount; rank++) {
    const flow = oracle.flows[rank];
    if (rank < count) {
      expect([
        result.ids[rank],
        result.originZones[rank],
        result.destinationZones[rank],
        result.counts[rank],
        result.weights[rank]
      ]).toEqual([flow.pairKey, flow.originZone, flow.destinationZone, flow.count, flow.weight]);
    } else {
      expect([
        result.ids[rank],
        result.originZones[rank],
        result.destinationZones[rank],
        result.counts[rank],
        result.weights[rank]
      ]).toEqual([NO_ZONE, NO_ZONE, NO_ZONE, 0, 0]);
    }
  }
  expectZoneTotals(result, oracle, hasWeights);
}

function expectZoneTotals(result: FlowResult, oracle: FlowOracleResult, hasWeights: boolean): void {
  expect(result.zoneOutCounts).toEqual(oracle.zoneOutCounts);
  expect(result.zoneInCounts).toEqual(oracle.zoneInCounts);
  if (hasWeights) {
    expect(result.zoneOutWeights).toEqual(oracle.zoneOutWeights);
    expect(result.zoneInWeights).toEqual(oracle.zoneInWeights);
  }
}

/** Seeded random OD zone IDs. */
function createZoneIds(seed: number, rowCount: number, zoneCount: number, outOfRange = 0) {
  const next = createSeededRandom(seed);
  const origins: number[] = [];
  const destinations: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    origins.push(Math.floor(next() * (zoneCount + outOfRange)));
    destinations.push(Math.floor(next() * (zoneCount + outOfRange)));
  }
  return {origins, destinations};
}

/** Integer weights 1..5 with a NaN every 17th row. */
function createWeights(seed: number, rowCount: number): number[] {
  const next = createSeededRandom(seed);
  return Array.from({length: rowCount}, (_, row) =>
    row % 17 === 5 ? Number.NaN : 1 + Math.floor(next() * 5)
  );
}

it('GPUFlowAggregation aggregates grid zones against the oracle (weighted and unweighted)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const bounds: [number, number, number, number] = [0, 0, 6, 5];
  const gridSize: [number, number] = [6, 5];
  const rowCount = 2000;
  const origins = createSeededPoints(11, rowCount, bounds);
  const destinations = createSeededPoints(23, rowCount, bounds);
  // Rejected rows: out of bounds, NaN, and exact maximum edges (which are accepted).
  origins.splice(0, 8, 7, 1, 1, -1, Number.NaN, 2, 3, Number.NaN);
  destinations.splice(10, 4, 6, 5, 0, 0);
  const weightValues = createWeights(5, rowCount);

  for (const useWeights of [true, false]) {
    for (const topCount of [50, 1024]) {
      const buffers = [
        createInputBuffer(device, Float32Array.from(origins)),
        createInputBuffer(device, Float32Array.from(destinations)),
        createInputBuffer(device, Float32Array.from(weightValues))
      ];
      const harness = createHarness(device, {
        id: 'flow-grid',
        zoneCount: 30,
        topCount,
        pairCapacity: 1024,
        hasWeights: useWeights,
        createProps: graph => ({
          zones: {kind: 'grid', bounds, gridSize},
          origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', rowCount),
          destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', rowCount),
          weights: useWeights
            ? importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
            : undefined
        })
      });
      harness.submit();
      const oracle = computeFlowAggregation({
        zones: {kind: 'grid', bounds, gridSize},
        origins,
        destinations,
        weights: useWeights ? weightValues : undefined
      });
      expect(oracle.flows.length).toBeGreaterThan(50);
      const result = await harness.read();
      expectMatchesOracle(result, oracle, topCount, useWeights);
      if (topCount === 50) {
        expect(result.overflow).toBe(1);
        expect(result.count).toBe(50);
        expect(result.requiredCount).toBe(oracle.flows.length);
      }
      harness.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('GPUFlowAggregation orders equal weights by pair key', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Pairs (origin, destination) with two rows each, inserted in scrambled order.
  const pairs = [
    [3, 1],
    [0, 2],
    [2, 2],
    [1, 3],
    [0, 2],
    [3, 1],
    [2, 2],
    [1, 3],
    [0, 0]
  ];
  const originIds = pairs.map(pair => pair[0]);
  const destinationIds = pairs.map(pair => pair[1]);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(originIds)),
    createInputBuffer(device, Uint32Array.from(destinationIds))
  ];
  const harness = createHarness(device, {
    id: 'flow-ties',
    zoneCount: 4,
    topCount: 8,
    pairCapacity: 8,
    hasWeights: false,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 4},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', pairs.length),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        pairs.length
      )
    })
  });
  harness.submit();
  const result = await harness.read();
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount: 4},
    originZoneIds: originIds,
    destinationZoneIds: destinationIds
  });
  expectMatchesOracle(result, oracle, 8, false);
  // Four pairs of count two ordered by key (0,2)=2, (1,3)=7, (2,2)=10, (3,1)=13, then (0,0).
  expect(result.ids.slice(0, 5)).toEqual([2, 7, 10, 13, 0]);
  expect(result.counts.slice(0, 5)).toEqual([2, 2, 2, 2, 1]);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

function createHexagonPoints(radii: number[]): number[] {
  const candidates = createSeededPoints(7, 200, [0, 0, 6, 6]);
  const points: number[] = [];
  for (let index = 0; index < candidates.length / 2 && points.length < 128; index++) {
    const [x, y] = [candidates[index * 2], candidates[index * 2 + 1]];
    const isClear = radii.every(radius => {
      const nearest = findNearestHexagon(x, y, 0, 0, radius, 10, 10);
      return nearest.secondNearest - nearest.nearest >= 1e-3 * radius;
    });
    if (isClear) {
      points.push(x, y);
    }
  }
  return [...points, Number.NaN, Number.NaN, 7, 7];
}

it('GPUFlowAggregation bins hexagon zones with per-frame radius and bounds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const origins = createHexagonPoints([1, 1.4, 0.6]);
  const rowCount = origins.length / 2;
  const destinations: number[] = [];
  for (let row = rowCount - 1; row >= 0; row--) {
    destinations.push(origins[row * 2], origins[row * 2 + 1]);
  }
  const weightValues = Array.from({length: rowCount}, (_, row) => 1 + (row % 4));
  const gridSize = getGPUPointDensityHexagonGridSize([0, 0, 6, 6], 1);
  const zoneCount = gridSize[0] * gridSize[1];
  const radius = new GPUParameterBuffer(device, {
    id: 'flow-radius',
    format: 'float32',
    length: 1,
    values: Float32Array.from([1])
  });
  const bounds = new GPUParameterBuffer(device, {
    id: 'flow-bounds',
    format: 'float32',
    length: 4,
    values: Float32Array.from([0, 0, 6, 6])
  });
  const buffers = [
    createInputBuffer(device, Float32Array.from(origins)),
    createInputBuffer(device, Float32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues))
  ];
  const harness = createHarness(device, {
    id: 'flow-hexagon',
    zoneCount,
    topCount: 64,
    pairCapacity: 512,
    hasWeights: true,
    createProps: graph => ({
      zones: {
        kind: 'hexagon',
        bounds: bounds.importToGraph(graph),
        gridSize,
        radius: radius.importToGraph(graph)
      },
      origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', rowCount),
      destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
    })
  });
  for (const [radiusValue, boundsValue] of [
    [1, [0, 0, 6, 6]],
    [1.4, [0, 0, 6, 6]],
    [0.6, [0, 0, 6, 6]],
    [1, [0, 0, 4, 4]]
  ] as const) {
    radius.write(Float32Array.from([radiusValue]));
    bounds.write(Float32Array.from(boundsValue));
    harness.submit();
    const oracle = computeFlowAggregation({
      zones: {
        kind: 'hexagon',
        bounds: [...boundsValue] as [number, number, number, number],
        gridSize,
        radius: radiusValue
      },
      origins,
      destinations,
      weights: weightValues
    });
    expectMatchesOracle(await harness.read(), oracle, 64, true);
  }
  harness.destroy();
  radius.destroy();
  bounds.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation applies mask, time window, and self-flow exclusion per frame', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 500;
  const zoneCount = 8;
  const {origins, destinations} = createZoneIds(3, rowCount, zoneCount, 2);
  const timestamps = Array.from({length: rowCount}, (_, row) => (row * 7) % 100);
  const weightValues = createWeights(9, rowCount);
  const maskValues = Array.from({length: rowCount}, (_, row) => (row % 3 === 0 ? 0 : 1));
  const window = new GPUParameterBuffer(device, {
    id: 'flow-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues({start: 10, end: 60})
  });
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues)),
    createInputBuffer(device, Uint32Array.from(maskValues)),
    createInputBuffer(device, Float32Array.from(timestamps))
  ];
  const harness = createHarness(device, {
    id: 'flow-ids',
    zoneCount,
    topCount: 64,
    pairCapacity: 64,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount),
      mask: importGraphBuffer(graph, 'mask', buffers[3], 'uint32', rowCount),
      timeWindow: {
        timestamps: importGraphBuffer(graph, 'timestamps', buffers[4], 'float32', rowCount),
        window: window.importToGraph(graph)
      },
      excludeSelfFlows: true
    })
  });
  const verify = async (start: number, end: number, mask: number[]) => {
    harness.submit();
    const oracle = computeFlowAggregation({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: origins,
      destinationZoneIds: destinations,
      weights: weightValues,
      mask,
      timeWindow: {timestamps, start, end},
      excludeSelfFlows: true
    });
    expectMatchesOracle(await harness.read(), oracle, 64, true);
    return oracle;
  };
  const first = await verify(10, 60, maskValues);
  expect(first.flows.length).toBeGreaterThan(0);
  window.write(getGPUTimeWindowFilterParameterValues({start: 40, end: 90}));
  const second = await verify(40, 90, maskValues);
  expect(second.flows).not.toEqual(first.flows);
  const newMask = maskValues.map(value => 1 - value);
  buffers[3].write(Uint32Array.from(newMask));
  await verify(40, 90, newMask);
  window.write(getGPUTimeWindowFilterParameterValues({start: 200, end: 100}));
  const empty = await verify(200, 100, newMask);
  expect(empty.flows.length).toBe(0);

  harness.destroy();
  window.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation reports pair table overflow and keeps zone totals exact', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 300;
  const zoneCount = 16;
  const {origins, destinations} = createZoneIds(41, rowCount, zoneCount);
  const weightValues = createWeights(2, rowCount);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues))
  ];
  const topCount = 6;
  const harness = createHarness(device, {
    id: 'flow-overflow',
    zoneCount,
    topCount,
    pairCapacity: 4,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
    })
  });
  harness.submit();
  const result = await harness.read();
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: weightValues
  });
  expect(oracle.flows.length).toBeGreaterThan(4);
  expect(result.pairOverflow).toBe(1);
  expect(result.overflow).toBe(1);
  expect(result.requiredCount).toBe(4);
  expect(result.count).toBe(4);
  expect(result.drawInstanceCount).toBe(4);
  for (let rank = 0; rank < topCount; rank++) {
    if (rank >= 4) {
      expect([result.ids[rank], result.counts[rank], result.weights[rank]]).toEqual([
        NO_ZONE,
        0,
        0
      ]);
      continue;
    }
    const flow = oracle.flows.find(candidate => candidate.pairKey === result.ids[rank]);
    expect(flow).toBeDefined();
    expect([result.counts[rank], result.weights[rank]]).toEqual([flow!.count, flow!.weight]);
    expect([result.originZones[rank], result.destinationZones[rank]]).toEqual([
      flow!.originZone,
      flow!.destinationZone
    ]);
    if (rank > 0) {
      const earlier = [result.weights[rank - 1], result.ids[rank - 1]];
      expect(
        earlier[0] > result.weights[rank] ||
          (earlier[0] === result.weights[rank] && earlier[1] < result.ids[rank])
      ).toBe(true);
    }
  }
  expectZoneTotals(result, oracle, true);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation handles empty input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers = [
    createInputBuffer(device, Uint32Array.from([0])),
    createInputBuffer(device, Uint32Array.from([0])),
    createInputBuffer(device, Float32Array.from([0]))
  ];
  const harness = createHarness(device, {
    id: 'flow-empty',
    zoneCount: 5,
    topCount: 4,
    pairCapacity: 8,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 5},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', 0),
      destinationZoneIds: importGraphBuffer(graph, 'destination-ids', buffers[1], 'uint32', 0),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', 0)
    })
  });
  harness.submit();
  const result = await harness.read();
  expectMatchesOracle(
    result,
    computeFlowAggregation({
      zones: {kind: 'ids', zoneCount: 5},
      originZoneIds: [],
      destinationZoneIds: []
    }),
    4,
    true
  );
  expect(result.count).toBe(0);
  expect(result.overflow).toBe(0);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation handles empty grid input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers = [
    createInputBuffer(device, Float32Array.from([0, 0])),
    createInputBuffer(device, Float32Array.from([0, 0]))
  ];
  const harness = createHarness(device, {
    id: 'flow-empty-grid',
    zoneCount: 4,
    topCount: 2,
    pairCapacity: 2,
    hasWeights: false,
    createProps: graph => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', 0),
      destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', 0)
    })
  });
  harness.submit();
  const result = await harness.read();
  expect(result.count).toBe(0);
  expect(result.overflow).toBe(0);
  expect(result.zoneOutCounts).toEqual([0, 0, 0, 0]);
  expect(result.ids).toEqual([NO_ZONE, NO_ZONE]);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation handles a single row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers = [
    createInputBuffer(device, Uint32Array.from([2])),
    createInputBuffer(device, Uint32Array.from([1])),
    createInputBuffer(device, Float32Array.from([7]))
  ];
  const harness = createHarness(device, {
    id: 'flow-single',
    zoneCount: 3,
    topCount: 4,
    pairCapacity: 1,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 3},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', 1),
      destinationZoneIds: importGraphBuffer(graph, 'destination-ids', buffers[1], 'uint32', 1),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', 1)
    })
  });
  harness.submit();
  const result = await harness.read();
  expectMatchesOracle(
    result,
    computeFlowAggregation({
      zones: {kind: 'ids', zoneCount: 3},
      originZoneIds: [2],
      destinationZoneIds: [1],
      weights: [7]
    }),
    4,
    true
  );
  expect(result.ids).toEqual([7, NO_ZONE, NO_ZONE, NO_ZONE]);
  expect(result.weights[0]).toBe(7);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation gates rows exactly with Int64 word timestamps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 900;
  const zoneCount = 6;
  const {origins, destinations} = createZoneIds(31, rowCount, zoneCount, 1);
  const weightValues = createWeights(13, rowCount);
  const TWO_32 = 2n ** 32n;
  // Windows across a low-word wrap (k * 2^32), at a real epoch (~1.7e12 ms), and high word zero.
  const windows = [
    {start: 3n * TWO_32 - 4n, end: 3n * TWO_32 + 6n},
    {start: 1_700_000_000_000n, end: 1_700_000_000_000n + 86_400_000n},
    {start: 395n * TWO_32 - 1000n, end: 396n * TWO_32 + 1000n},
    {start: 100n, end: 100n},
    {start: 50n, end: 10n}
  ];
  const timestamps: bigint[] = [];
  for (const {start, end} of windows) {
    for (const edge of [start, end]) {
      timestamps.push(edge - 1n, edge, edge + 1n);
    }
  }
  const next = createSeededRandom(77);
  while (timestamps.length < rowCount) {
    const {start, end} = windows[Math.floor(next() * windows.length)];
    timestamps.push(start - 2000n + BigInt(Math.floor(next() * Number(end - start + 4000n))));
  }
  const timestampWords = getInt64TimeWords(BigInt64Array.from(timestamps));
  const window = new GPUParameterBuffer(device, {
    id: 'flow-word-window',
    format: 'uint32',
    length: GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterWordParameterValues(windows[0])
  });
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues)),
    createInputBuffer(device, timestampWords)
  ];
  const harness = createHarness(device, {
    id: 'flow-word-time',
    zoneCount,
    topCount: 64,
    pairCapacity: 64,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount),
      timeWindow: {
        timestamps: importGraphBuffer(graph, 'timestamps', buffers[3], 'uint32x2', rowCount),
        window: window.importToGraph(graph)
      }
    })
  });
  let previousTotal = -1;
  for (const {start, end} of windows) {
    window.write(getGPUTimeWindowFilterWordParameterValues({start, end}));
    harness.submit();
    const oracle = computeFlowAggregation({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: origins,
      destinationZoneIds: destinations,
      weights: weightValues,
      wordTimeWindow: {timestamps, start, end}
    });
    const result = await harness.read();
    expectMatchesOracle(result, oracle, 64, true);
    const accepted = result.zoneOutCounts.reduce((sum, value) => sum + value, 0);
    if (end >= start) {
      expect(accepted).toBeGreaterThan(0);
      expect(accepted).toBeLessThan(rowCount);
    } else {
      expect(accepted).toBe(0);
    }
    expect(accepted).not.toBe(previousTotal);
    previousTotal = accepted;
  }
  harness.destroy();
  window.destroy();
  for (const buffer of buffers) buffer.destroy();
});

/** Weights that expose summation order: 1e8, 1, fractions, and non-finite values. */
function createOrderSensitiveWeights(seed: number, rowCount: number): Float32Array {
  const next = createSeededRandom(seed);
  return Float32Array.from({length: rowCount}, (_, row) => {
    if (row % 53 === 7) return Number.NaN;
    const draw = next();
    if (draw < 0.3) return 1e8;
    if (draw < 0.6) return 1;
    return Math.fround(next() * 3.3);
  });
}

function getFloat32Bits(values: readonly number[]): number[] {
  return Array.from(new Uint32Array(Float32Array.from(values).buffer));
}

it('GPUFlowAggregation sorted sums are bitwise stable and match the fixed-tree oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 4000;
  const zoneCount = 3;
  const topCount = 16;
  const {origins, destinations} = createZoneIds(41, rowCount, zoneCount, 1);
  const weightValues = createOrderSensitiveWeights(19, rowCount);
  const weightArray = Array.from(weightValues);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  const createFlowHarness = (sumOrder: 'atomic' | 'sorted') =>
    createHarness(device, {
      id: `flow-${sumOrder}-sums`,
      zoneCount,
      topCount,
      pairCapacity: 16,
      hasWeights: true,
      createProps: graph => ({
        sumOrder,
        zones: {kind: 'ids', zoneCount},
        originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
        destinationZoneIds: importGraphBuffer(
          graph,
          'destination-ids',
          buffers[1],
          'uint32',
          rowCount
        ),
        weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
      })
    });
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: weightArray,
    sumMode: 'tree'
  });
  const sequentialOracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: weightArray
  });

  const sorted = createFlowHarness('sorted');
  let firstBits: number[] | undefined;
  for (let run = 0; run < 5; run++) {
    sorted.submit();
    const result = await sorted.read();
    expectMatchesOracle(result, oracle, topCount, true);
    // Tight agreement with the float64 sequential sum.
    for (let rank = 0; rank < result.count; rank++) {
      const exact = sequentialOracle.flows.find(flow => flow.pairKey === result.ids[rank])!;
      expect(Math.abs(result.weights[rank] - exact.weight)).toBeLessThanOrEqual(
        Math.abs(exact.weight) * 1e-3 + 64
      );
    }
    const bits = getFloat32Bits([
      ...result.weights,
      ...result.zoneOutWeights,
      ...result.zoneInWeights
    ]);
    firstBits ??= bits;
    expect(bits).toEqual(firstBits);
  }
  sorted.destroy();

  // Informational: how far the atomic sums are from the fixed-order sums for this input.
  const atomic = createFlowHarness('atomic');
  let differing = 0;
  for (let run = 0; run < 3; run++) {
    atomic.submit();
    const result = await atomic.read();
    differing += getFloat32Bits(result.zoneOutWeights).filter(
      (bits, zone) => bits !== getFloat32Bits(oracle.zoneOutWeights)[zone]
    ).length;
    expect(result.zoneOutCounts).toEqual(oracle.zoneOutCounts);
  }
  console.log(
    `flow sums: atomic zoneOutWeights differing from fixed-tree bits over 3 runs: ${differing}`
  );
  atomic.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation sorted sums keep zone totals exact when the pair table overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 3000;
  const zoneCount = 40;
  const {origins, destinations} = createZoneIds(5, rowCount, zoneCount, 2);
  const weightValues = createOrderSensitiveWeights(23, rowCount);
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: Array.from(weightValues),
    sumMode: 'tree'
  });
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  const harness = createHarness(device, {
    id: 'flow-sorted-overflow',
    zoneCount,
    topCount: 8,
    pairCapacity: 64,
    maximumProbeCount: 8,
    hasWeights: true,
    createProps: graph => ({
      sumOrder: 'sorted',
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
    })
  });
  harness.submit();
  const result = await harness.read();
  expect(result.pairOverflow).toBe(1);
  expect(result.overflow).toBe(1);
  expectZoneTotals(result, oracle, true);
  // Reported flow sums are each pair's fixed-tree sum, in descending order.
  for (let rank = 0; rank < result.count; rank++) {
    const flow = oracle.flows.find(candidate => candidate.pairKey === result.ids[rank])!;
    expect(result.weights[rank]).toBe(flow.weight);
    if (rank > 0) expect(result.weights[rank]).toBeLessThanOrEqual(result.weights[rank - 1]);
  }
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation sorted sums with grid zones and a mask match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const bounds: [number, number, number, number] = [0, 0, 3, 2];
  const gridSize: [number, number] = [3, 2];
  const rowCount = 1500;
  const origins = createSeededPoints(3, rowCount, bounds);
  const destinations = createSeededPoints(4, rowCount, bounds);
  origins.splice(0, 4, 9, 9, Number.NaN, 1);
  const weightValues = createOrderSensitiveWeights(2, rowCount);
  const maskValues = Array.from({length: rowCount}, (_, row) => (row % 5 === 0 ? 0 : 1));
  const buffers = [
    createInputBuffer(device, Float32Array.from(origins)),
    createInputBuffer(device, Float32Array.from(destinations)),
    createInputBuffer(device, weightValues),
    createInputBuffer(device, Uint32Array.from(maskValues))
  ];
  const harness = createHarness(device, {
    id: 'flow-sorted-grid',
    zoneCount: 6,
    topCount: 36,
    pairCapacity: 64,
    hasWeights: true,
    createProps: graph => ({
      sumOrder: 'sorted',
      zones: {kind: 'grid', bounds, gridSize},
      origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', rowCount),
      destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount),
      mask: importGraphBuffer(graph, 'mask', buffers[3], 'uint32', rowCount)
    })
  });
  harness.submit();
  expectMatchesOracle(
    await harness.read(),
    computeFlowAggregation({
      zones: {kind: 'grid', bounds, gridSize},
      origins,
      destinations,
      weights: Array.from(weightValues),
      mask: maskValues,
      sumMode: 'tree'
    }),
    36,
    true
  );
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation measures atomic versus sorted sum cost on 1M rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 1 << 20;
  const zoneCount = 64;
  const {origins, destinations} = createZoneIds(8, rowCount, zoneCount);
  const weightValues = createOrderSensitiveWeights(6, rowCount);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const harness = createHarness(device, {
      id: `flow-cost-${sumOrder}`,
      zoneCount,
      topCount: 64,
      pairCapacity: 8192,
      maximumProbeCount: 64,
      hasWeights: true,
      createProps: graph => ({
        sumOrder,
        zones: {kind: 'ids', zoneCount},
        originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
        destinationZoneIds: importGraphBuffer(
          graph,
          'destination-ids',
          buffers[1],
          'uint32',
          rowCount
        ),
        weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
      })
    });
    harness.submit();
    await harness.read();
    const runCount = 3;
    const startTime = performance.now();
    for (let run = 0; run < runCount; run++) {
      harness.submit();
      await harness.read();
    }
    const milliseconds = (performance.now() - startTime) / runCount;
    const result = await harness.read();
    expect(result.zoneOutCounts.reduce((sum, value) => sum + value, 0)).toBe(rowCount);
    console.log(`flow cost 1M rows, ${sumOrder}: ${milliseconds.toFixed(1)} ms per submit+read`);
    harness.destroy();
  }
  for (const buffer of buffers) buffer.destroy();
});

it('aggregates lexicographic flow keys above 65535 zones without uint32 collisions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const origins = new Uint32Array([
    65536,
    0,
    65536,
    32768,
    0,
    65536,
    131071,
    65536,
    131072,
    NO_ZONE
  ]);
  const destinations = new Uint32Array([1, 1, 2, 1, 1, 1, 65536, 1, 0, 0]);
  const inputWeights = new Float32Array([2, 5, -1, 10, 5, 0, 10, 0, 999, 999]);
  const buffers = [origins, destinations, inputWeights].map(values =>
    createInputBuffer(device, values)
  );
  try {
    for (const hasWeights of [false, true]) {
      for (const topCount of [3, 6]) {
        const fixture = createHarness(device, {
          id: `large-zone-flows-${hasWeights}-${topCount}`,
          zoneCount: 131072,
          topCount,
          pairCapacity: 32,
          hasWeights,
          createProps: graph => ({
            zones: {kind: 'ids', zoneCount: 131072},
            originZoneIds: importGraphBuffer(
              graph,
              'large-origin-input',
              buffers[0],
              'uint32',
              origins.length
            ),
            destinationZoneIds: importGraphBuffer(
              graph,
              'large-destination-input',
              buffers[1],
              'uint32',
              origins.length
            ),
            weights: hasWeights
              ? importGraphBuffer(
                  graph,
                  'large-weight-input',
                  buffers[2],
                  'float32',
                  origins.length
                )
              : undefined
          })
        });
        try {
          // Closed-form groups. (0,1), (32768,1) and (65536,1) collided in uint32 packed keys.
          const expectedOrigins = hasWeights
            ? [0, 32768, 131071, 65536, 65536]
            : [65536, 0, 32768, 65536, 131071];
          const expectedDestinations = hasWeights ? [1, 1, 65536, 1, 2] : [1, 1, 1, 2, 65536];
          const expectedIds = hasWeights ? [1, 3, 6, 0, 2] : [0, 1, 3, 2, 6];
          const expectedCounts = hasWeights ? [2, 1, 1, 3, 1] : [3, 2, 1, 1, 1];
          const expectedWeights = hasWeights ? [10, 10, 10, 2, -1] : expectedCounts;
          for (let encoding = 0; encoding < 2; encoding++) {
            fixture.submit();
            const result = await fixture.read();
            const count = Math.min(topCount, 5);
            const pad = (values: number[], value: number) => [
              ...values.slice(0, count),
              ...Array(topCount - count).fill(value)
            ];
            expect(result.ids).toEqual(pad(expectedIds, NO_ZONE));
            expect(result.originZones).toEqual(pad(expectedOrigins, NO_ZONE));
            expect(result.destinationZones).toEqual(pad(expectedDestinations, NO_ZONE));
            expect(result.counts).toEqual(pad(expectedCounts, 0));
            expect(result.weights).toEqual(pad(expectedWeights, 0));
            expect(result.count).toBe(count);
            expect(result.requiredCount).toBe(5);
            expect(result.pairOverflow).toBe(0);
            expect(result.overflow).toBe(topCount < 5 ? 1 : 0);
            expect(result.zoneOutCounts.reduce((sum, value) => sum + value, 0)).toBe(8);
            expect(result.zoneInCounts[1]).toBe(6);
            expect(result.zoneOutCounts[65536]).toBe(4);
          }
        } finally {
          fixture.destroy();
        }
      }
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

it('keeps pair-capacity overflow separate from top-K counts with large zone IDs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const origins = createInputBuffer(device, new Uint32Array([0, 32768, 65536, 131071]));
  const destinations = createInputBuffer(device, new Uint32Array([1, 1, 1, 1]));
  const fixture = createHarness(device, {
    id: 'large-zone-pair-overflow',
    zoneCount: 131072,
    topCount: 4,
    pairCapacity: 2,
    hasWeights: false,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 131072},
      originZoneIds: importGraphBuffer(graph, 'overflow-origins', origins, 'uint32', 4),
      destinationZoneIds: importGraphBuffer(
        graph,
        'overflow-destinations',
        destinations,
        'uint32',
        4
      )
    })
  });
  try {
    fixture.submit();
    const result = await fixture.read();
    expect(result.requiredCount).toBe(2);
    expect(result.count).toBe(2);
    expect(result.pairOverflow).toBe(1);
    expect(result.overflow).toBe(1);
    expect(result.zoneInCounts[1]).toBe(4);
    expect(result.counts.slice(0, 2)).toEqual([1, 1]);
  } finally {
    fixture.destroy();
    origins.destroy();
    destinations.destroy();
  }
});

it('flow chunk parameters preserve row bases, source offsets and unequal chunk boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const buffers: Buffer[] = [];
  const fixture = createHarness(device, {
    id: 'flow-chunk-parameters',
    zoneCount: 4,
    topCount: 4,
    pairCapacity: 16,
    hasWeights: true,
    createProps: graph => {
      const positions = [0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.25, 0.75];
      const destinations = [0.75, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75];
      const chunks = <Format extends 'float32' | 'float32x2'>(
        name: string,
        format: Format,
        values: number[],
        lengths: number[]
      ) => {
        const components = format === 'float32x2' ? 2 : 1;
        let rowStart = 0;
        return createVectorView(
          name,
          format,
          lengths.map((length, index) => {
            const padding = index + 1;
            const data = new Float32Array((padding + length) * components).fill(999);
            data.set(
              values.slice(rowStart * components, (rowStart + length) * components),
              padding * components
            );
            rowStart += length;
            const buffer = createInputBuffer(device, data);
            buffers.push(buffer);
            const backing = importGraphBuffer(
              graph,
              `${name}-${index}`,
              buffer,
              format,
              padding + length
            );
            return graph.createDataView(backing.buffer, {
              format,
              byteOffset: padding * components * 4,
              length
            });
          })
        );
      };
      return {
        zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
        origins: chunks('unequal-origins', 'float32x2', positions, [1, 3, 2]),
        destinations: chunks('unequal-destinations', 'float32x2', destinations, [2, 1, 3]),
        weights: chunks('unequal-weights', 'float32', [1, 2, 3, 4, 5, 6], [3, 1, 2])
      };
    }
  });
  try {
    for (let encoding = 0; encoding < 2; encoding++) {
      fixture.submit();
      const result = await fixture.read();
      expect(result.ids).toEqual([11, 4, 1, NO_ZONE]);
      expect(result.counts).toEqual([2, 2, 2, 0]);
      expect(result.weights).toEqual([11, 7, 3, 0]);
      expect(result.originZones).toEqual([2, 1, 0, NO_ZONE]);
      expect(result.destinationZones).toEqual([3, 0, 1, NO_ZONE]);
      expect(result.requiredCount).toBe(3);
      expect(result.overflow).toBe(0);
    }
  } finally {
    fixture.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

/** Compact-only harness: huge zone IDs never require a zone-sized CPU or GPU allocation. */
function createSparseFlowReviewHarness(
  device: Device,
  origins: Uint32Array,
  destinations: Uint32Array,
  inputWeights: Float32Array,
  capacity: number,
  sumOrder: 'sorted' | 'atomic',
  columnStride?: number
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, values.length);
  };
  const outputs: {
    name: string;
    buffer: Buffer;
    offset: number;
    length: number;
    format: 'uint32' | 'float32';
  }[] = [];
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.push({name, buffer, offset: 0, length, format});
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const ids = output('ids', 'uint32', capacity);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const requiredCount = output('requiredCount', 'uint32', 1);
  const pairOverflow = output('pairOverflow', 'uint32', 1);
  let originOutput: GraphDataView<'uint32'>;
  let destinationOutput: GraphDataView<'uint32'>;
  if (columnStride === undefined) {
    originOutput = output('origins', 'uint32', capacity);
    destinationOutput = output('destinations', 'uint32', capacity);
  } else {
    const buffer = createOutputBuffer(device, columnStride / 4 + capacity);
    buffers.push(buffer);
    const backing = importGraphBuffer(
      graph,
      'shared-columns',
      buffer,
      'uint32',
      columnStride / 4 + capacity
    );
    originOutput = graph.createDataView(backing.buffer, {format: 'uint32', length: capacity});
    destinationOutput = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: columnStride,
      length: capacity
    });
    outputs.push(
      {name: 'origins', buffer, offset: 0, length: capacity, format: 'uint32'},
      {name: 'destinations', buffer, offset: columnStride / 4, length: capacity, format: 'uint32'}
    );
  }
  const counts = output('counts', 'uint32', capacity);
  const weights = output('weights', 'float32', capacity);
  let compiled: CompiledGPUCommandGraph<void>;
  try {
    graph.add(
      new GPUFlowAggregation({
        zones: {kind: 'ids', zoneCount: 0xfffffffe},
        pairCapacity: 32,
        sumOrder,
        originZoneIds: input('source-origins', 'uint32', origins),
        destinationZoneIds: input('source-destinations', 'uint32', destinations),
        weights: input('source-weights', 'float32', inputWeights),
        output: {
          ids,
          status: {count, overflow, requiredCount, pairOverflow},
          flowOriginZoneIds: originOutput,
          flowDestinationZoneIds: destinationOutput,
          flowCounts: counts,
          flowWeights: weights
        }
      })
    );
    compiled = graph.compile();
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    throw error;
  }
  return {
    async run(): Promise<Record<string, number[]>> {
      for (const entry of outputs)
        entry.buffer.write(new Uint32Array(entry.offset + entry.length).fill(GARBAGE));
      submitGraph(device, compiled, undefined);
      const result: Record<string, number[]> = {};
      for (const entry of outputs) {
        const values =
          entry.format === 'uint32'
            ? await readUint32(entry.buffer, entry.offset + entry.length)
            : await readFloat32(entry.buffer, entry.offset + entry.length);
        result[entry.name] = values.slice(entry.offset);
      }
      return result;
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it('large unsigned zone tuples cross local, workgroup and radix boundaries with stable ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  for (const runLengths of [
    [18, 13],
    [18, 240, 257, 7]
  ]) {
    const tuples = [
      [0x80000000, 1],
      [0xfffffffd, 0x80000000],
      [1, 0xfffffffd],
      [0x80000000, 0xfffffffd]
    ];
    const origins: number[] = [],
      destinations: number[] = [],
      weights: number[] = [],
      representatives: number[] = [];
    for (const [group, length] of runLengths.entries()) {
      representatives.push(origins.length);
      for (let row = 0; row < length; row++) {
        origins.push(tuples[group][0]);
        destinations.push(tuples[group][1]);
        weights.push(row === 0 ? 1 : 0);
      }
    }
    // Both missing keys and the first ID outside the accepted range must reject their rows.
    origins.push(NO_ZONE, 0xfffffffe, 0);
    destinations.push(0, 1, NO_ZONE);
    weights.push(999, 999, 999);
    const order = runLengths.length === 2 ? [0, 1] : [2, 0, 3, 1];
    for (const capacity of [1, 6])
      for (const sumOrder of ['sorted', 'atomic'] as const) {
        const fixture = createSparseFlowReviewHarness(
          device,
          Uint32Array.from(origins),
          Uint32Array.from(destinations),
          Float32Array.from(weights),
          capacity,
          sumOrder
        );
        try {
          const count = Math.min(capacity, runLengths.length);
          const pad = (values: number[], empty: number) => [
            ...values.slice(0, count),
            ...Array(capacity - count).fill(empty)
          ];
          const expected = {
            ids: pad(
              order.map(group => representatives[group]),
              NO_ZONE
            ),
            origins: pad(
              order.map(group => tuples[group][0]),
              NO_ZONE
            ),
            destinations: pad(
              order.map(group => tuples[group][1]),
              NO_ZONE
            ),
            counts: pad(
              order.map(group => runLengths[group]),
              0
            ),
            weights: pad(
              order.map(() => 1),
              0
            ),
            count: [count],
            requiredCount: [runLengths.length],
            overflow: [capacity < runLengths.length ? 1 : 0],
            pairOverflow: [0]
          };
          expect(await fixture.run()).toEqual(expected);
          expect(await fixture.run()).toEqual(expected);
        } finally {
          fixture.destroy();
        }
      }
  }
});

it('flow shared output buffers reject adjacent slices and accept independent aligned bindings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const origins = new Uint32Array([0x80000000, 0x80000000]);
  const destinations = new Uint32Array([0xfffffffd, 0xfffffffd]);
  const weights = new Float32Array([0.25, 0.5]);
  expect(() =>
    createSparseFlowReviewHarness(device, origins, destinations, weights, 1, 'sorted', 4)
  ).toThrow(/aligned binding byte ranges/);
  const fixture = createSparseFlowReviewHarness(
    device,
    origins,
    destinations,
    weights,
    1,
    'sorted',
    256
  );
  try {
    expect(await fixture.run()).toEqual({
      ids: [0],
      origins: [0x80000000],
      destinations: [0xfffffffd],
      counts: [2],
      weights: [0.75],
      count: [1],
      requiredCount: [1],
      overflow: [0],
      pairOverflow: [0]
    });
  } finally {
    fixture.destroy();
  }
});

it('packed and unequal empty flow chunks beyond 256-byte offsets match every CPU-oracle output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const rowCount = 522;
  const originIds = Array.from({length: rowCount}, (_, row) => row % 4);
  const destinationIds = Array.from({length: rowCount}, (_, row) => (row * 3 + 1) % 4);
  const positions = (ids: number[]) =>
    ids.flatMap(zone => [zone % 2 ? 0.75 : 0.25, zone >= 2 ? 0.75 : 0.25]);
  const originPositions = positions(originIds),
    destinationPositions = positions(destinationIds);
  const weights = createOrderSensitiveWeights(137, rowCount);
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount: 4},
    originZoneIds: originIds,
    destinationZoneIds: destinationIds,
    weights: Array.from(weights),
    sumMode: 'tree'
  });
  let packed: FlowResult | undefined;
  for (const chunked of [false, true]) {
    const buffers: Buffer[] = [];
    const fixture = createHarness(device, {
      id: `flow-empty-offset-${chunked}`,
      zoneCount: 4,
      topCount: 6,
      pairCapacity: 16,
      hasWeights: true,
      createProps: graph => {
        const chunks = <Format extends 'float32' | 'float32x2'>(
          name: string,
          format: Format,
          data: ArrayLike<number>,
          lengths: number[]
        ) => {
          const components = format === 'float32x2' ? 2 : 1;
          let rowBase = 0;
          const views = lengths.map((length, index) => {
            const padding = 70 + index;
            const values = new Float32Array((padding + length) * components).fill(999);
            for (let component = 0; component < length * components; component++)
              values[padding * components + component] = data[rowBase * components + component];
            rowBase += length;
            const buffer = createInputBuffer(device, values);
            buffers.push(buffer);
            const backing = importGraphBuffer(
              graph,
              `${name}-${index}`,
              buffer,
              format,
              padding + length
            );
            return graph.createDataView(backing.buffer, {
              format,
              byteOffset: padding * components * 4,
              length
            });
          });
          return chunked ? createVectorView(name, format, views) : views[0];
        };
        return {
          sumOrder: 'sorted',
          zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
          origins: chunks(
            'review-origins',
            'float32x2',
            originPositions,
            chunked ? [0, 1, 17, 0, 240, 264, 0] : [rowCount]
          ),
          destinations: chunks(
            'review-destinations',
            'float32x2',
            destinationPositions,
            chunked ? [16, 0, 240, 1, 0, 255, 10] : [rowCount]
          ),
          weights: chunks(
            'review-weights',
            'float32',
            weights,
            chunked ? [0, 3, 256, 0, 1, 262, 0] : [rowCount]
          )
        };
      }
    });
    try {
      for (let encoding = 0; encoding < 2; encoding++) {
        fixture.submit();
        const result = await fixture.read();
        expectMatchesOracle(result, oracle, 6, true);
        expect(result.zoneOutCounts).toEqual([131, 131, 130, 130]);
        if (packed) expect(result).toEqual(packed);
        else packed = result;
      }
    } finally {
      fixture.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('skewed flow chunks launch their own workgroups and preserve closed-form aggregates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const lengths = [1000000, ...Array(1023).fill(1)];
  const rowCount = 1001023;
  const buffers: Buffer[] = [];
  const fixture = createHarness(device, {
    id: 'flow-skewed-dispatch',
    zoneCount: 4,
    topCount: 2,
    pairCapacity: 8,
    maximumProbeCount: 1,
    hasWeights: true,
    createProps: graph => {
      const origins = lengths.map((length, index) => {
        const values = new Float32Array(length * 2).fill(0.25);
        const buffer = createInputBuffer(device, values);
        buffers.push(buffer);
        return importGraphBuffer(graph, `skewed-origin-${index}`, buffer, 'float32x2', length);
      });
      const weights = lengths.map((length, index) => {
        const buffer = createInputBuffer(device, new Float32Array(length).fill(0.125));
        buffers.push(buffer);
        return importGraphBuffer(graph, `skewed-weight-${index}`, buffer, 'float32', length);
      });
      const destinations = new Float32Array(rowCount * 2);
      for (let row = 0; row < rowCount; row++) {
        destinations[row * 2] = 0.75;
        destinations[row * 2 + 1] = 0.25;
      }
      const buffer = createInputBuffer(device, destinations);
      buffers.push(buffer);
      return {
        zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
        origins: createVectorView('skewed-origin-vector', 'float32x2', origins),
        destinations: importGraphBuffer(graph, 'skewed-destination', buffer, 'float32x2', rowCount),
        weights: createVectorView('skewed-weight-vector', 'float32', weights)
      };
    }
  });
  try {
    for (const pattern of [/origin-zones/, /contributions/]) {
      const nodes = fixture.compiled.preflight.nodes.filter(node => pattern.test(node.id));
      expect(nodes.reduce((total, node) => total + node.maximumInvocationCount, 0)).toBe(rowCount);
      expect(nodes.reduce((total, node) => total + node.maximumWorkgroupCount, 0)).toBe(4930);
    }
    const started = performance.now();
    fixture.submit();
    const result = await fixture.read();
    console.log(
      `flow skewed chunks: 4930 workgroups per chunked pass; ${performance.now() - started} ms submit/read`
    );
    expect(result).toEqual({
      ids: [1, NO_ZONE],
      count: 1,
      requiredCount: 1,
      overflow: 0,
      pairOverflow: 0,
      drawInstanceCount: 1,
      originZones: [0, NO_ZONE],
      destinationZones: [1, NO_ZONE],
      counts: [rowCount, 0],
      weights: [rowCount / 8, 0],
      zoneOutCounts: [rowCount, 0, 0, 0],
      zoneInCounts: [0, rowCount, 0, 0],
      zoneOutWeights: [rowCount / 8, 0, 0, 0],
      zoneInWeights: [0, rowCount / 8, 0, 0]
    });
  } finally {
    fixture.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('finite float32 literal bounds compile and preserve a closed-form grid result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const buffer = createInputBuffer(device, new Float32Array([0, 0, 0, 0]));
  const fixture = createHarness(device, {
    id: 'flow-f32-literal-limit',
    zoneCount: 1,
    topCount: 1,
    pairCapacity: 8,
    hasWeights: false,
    createProps: graph => {
      const positions = importGraphBuffer(graph, 'literal-limit-positions', buffer, 'float32x2', 2);
      return {
        zones: {
          kind: 'grid',
          bounds: [0, 0, 3.4028234663852886e38, 3.4028234663852886e38],
          gridSize: [1, 1]
        },
        origins: positions,
        destinations: positions
      };
    }
  });
  try {
    fixture.submit();
    const result = await fixture.read();
    expect(result.ids).toEqual([0]);
    expect(result.counts).toEqual([2]);
    expect(result.weights).toEqual([2]);
    expect(result.originZones).toEqual([0]);
    expect(result.destinationZones).toEqual([0]);
    expect(result.zoneOutCounts).toEqual([2]);
    expect(result.zoneInCounts).toEqual([2]);
    expect(result.requiredCount).toBe(1);
    expect(result.overflow).toBe(0);
  } finally {
    fixture.destroy();
    buffer.destroy();
  }
});

it('hexagon signed cell clamp values remain outside the largest supported axis', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (!device) return;
  const graph = new GPUCommandGraph<void>(device);
  const positionsBuffer = createInputBuffer(device, new Float32Array([0, 0, 1e20, 0]));
  const positions = importGraphBuffer(
    graph,
    'signed-clamp-positions',
    positionsBuffer,
    'float32x2',
    2
  );
  const outputBuffers: Buffer[] = [];
  const outputs = new Map<string, Buffer>();
  const output = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    outputBuffers.push(buffer);
    outputs.set(name, buffer);
    return importGraphBuffer(graph, `signed-clamp-${name}`, buffer, 'uint32', 1);
  };
  let compiled: CompiledGPUCommandGraph<void> | undefined;
  try {
    graph.add(
      new GPUFlowAggregation({
        zones: {kind: 'hexagon', bounds: [0, 0, 1e20, 1], gridSize: [2147483520, 1], radius: 1},
        origins: positions,
        destinations: positions,
        pairCapacity: 8,
        output: {
          ids: output('ids'),
          status: {
            count: output('count'),
            overflow: output('overflow'),
            requiredCount: output('requiredCount'),
            pairOverflow: output('pairOverflow')
          },
          flowOriginZoneIds: output('origin'),
          flowDestinationZoneIds: output('destination'),
          flowCounts: output('counts')
        }
      })
    );
    compiled = graph.compile();
    for (let repeat = 0; repeat < 2; repeat++) {
      submitGraph(device, compiled, undefined);
      const result: Record<string, number[]> = {};
      for (const [name, buffer] of outputs) result[name] = await readUint32(buffer, 1);
      expect(result).toEqual({
        ids: [0],
        count: [1],
        overflow: [0],
        requiredCount: [1],
        pairOverflow: [0],
        origin: [0],
        destination: [0],
        counts: [1]
      });
    }
  } finally {
    compiled?.destroy();
    positionsBuffer.destroy();
    for (const buffer of outputBuffers) buffer.destroy();
  }
});
