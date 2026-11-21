// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  GPUSpatialScanStatistic,
  createSpatialScanStatisticForTesting
} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {GPUExpectationBasedSpaceTimeScan} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {getScanBoundaryPopulationWGSL} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-nearest-zones';
import {
  getScanNearestZoneSearchPlan,
  type GPUScanNearestZoneSearch
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-nearest-index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';

// The Poisson shader writes a rounded population, a boundary count and ten exact limbs
// per zone after the separation keys. The interpreter requires space for every write.
const POISSON_BOUNDARY_WORDS_PER_ZONE = 12;

it('one-zone scan indexes short lists instead of evaluating every pair four times', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scan-index-${serial++}`, format, length);
  const zones = 64;
  try {
    const scan = new GPUSpatialScanStatistic({
      positions: view('float32x2', zones),
      cases: view('uint32', zones),
      baseline: view('float32', zones),
      timeBuckets: 1,
      maximumWindowZones: 1,
      maximumPermutations: 1,
      maximumClusters: 1,
      parameters: view('uint32', 8),
      output: {
        clusterIndices: view('uint32', 4),
        clusterStatistics: view('float32', 8),
        statistics: view('float32', 2),
        summary: view('uint32', 4)
      }
    });
    const kernel = getGPUCommandNodeKernelSources(scan.getCommandNodes(graph)).find(
      kernel => kernel.id === 'scan-statistic-nearest'
    )!;
    const positions = Float32Array.from({length: zones * 2}, (_, component) =>
      component % 2 === 0 ? component / 2 : 0
    );
    const memory: Record<string, Uint32Array | Float32Array> = {
      positions,
      neighbors: new Uint32Array(zones * 2),
      distances: new Uint32Array(zones * (2 + POISSON_BOUNDARY_WORDS_PER_ZONE)),
      population: new Float32Array(zones).fill(1),
      sortKeys: new Uint32Array(zones * 2),
      sortZones: new Uint32Array(zones * 2),
      sortedRowsX: Uint32Array.from({length: zones}, (_, row) => row),
      sortedRowsY: Uint32Array.from({length: zones}, (_, row) => row),
      sortedKeysX: Uint32Array.from(
        {length: zones},
        (_, row) => new Uint32Array(Float32Array.of(row).buffer)[0] ^ 0x80000000
      ),
      sortedKeysY: new Uint32Array(zones).fill(0x80000000)
    };
    const counts = new Uint32Array(zones);
    const bindings: Record<number, Record<number, Uint32Array | Float32Array>> = {0: {8: counts}};
    for (const binding of new WgslReflect(kernel.source).storage)
      bindings[binding.group][binding.binding] = memory[binding.name];
    // Instrument production separation arithmetic; no alternate nearest implementation is run.
    const source = `var<private> keyEvaluationCount: u32;
@group(0) @binding(8) var<storage, read_write> keyCounts: array<u32>;
${kernel.source}`
      .replace(
        /fn getSeparationKey\(([^)]*)\) -> f32 \{/,
        'fn getSeparationKey($1) -> f32 { keyEvaluationCount++;'
      )
      .replace('let row = index * LIST;', 'let row = index * LIST; keyEvaluationCount = 0u;')
      .replace(/^\s*_ = [^;]+;/gm, '')
      .replace(/}\s*$/, 'keyCounts[index] = keyEvaluationCount; }');
    const workgroupSize = Number(source.match(/@workgroup_size\((\d+)\)/)![1]);
    // WgslExec leaks invocation-local state between threads; isolate queries and keep their arithmetic intact.
    const isolatedSource = `override QUERY_ROW: u32;\n${source}`
      .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
      .replace(
        `let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;`,
        'let index = QUERY_ROW;'
      );
    for (let query = 0; query < zones; query++)
      new WgslExec(new WgslParser().parse(isolatedSource)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        bindings,
        {
          constants: {
            QUERY_ROW: query,
            INVOCATION_COUNT: zones,
            WGSL_KERNEL_DISPATCH_X: 1,
            WGSL_KERNEL_DISPATCH_Y: 1
          }
        }
      );
    expect(Math.max(...counts)).toBeLessThanOrEqual(16);
    for (let row = 0; row < zones; row++) {
      expect(memory['neighbors'][row * 2]).toBe(row);
      expect(memory['neighbors'][row * 2 + 1]).toBe(row === 0 ? 1 : row - 1);
      const boundaryPopulation = new Float32Array(memory['distances'].buffer)[zones * 2 + row];
      expect(boundaryPopulation).toBe(row === 0 || row === zones - 1 ? 2 : 3);
    }
  } finally {
    device.destroy();
  }
});

function createScene(
  zones: number,
  listLength: number,
  options: {
    geographic?: boolean;
    nearestZoneSearch?: GPUScanNearestZoneSearch;
    expectationBased?: boolean;
  } = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scan-index-${serial++}`, format, length);
  const common = {
    id: 'scan-index',
    positions: view('float32x2', zones),
    cases: view('uint32', zones),
    baseline: view('float32', zones),
    spatialContext: options.geographic
      ? ({coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const)
      : undefined,
    nearestZoneSearch: options.nearestZoneSearch,
    timeBuckets: 1,
    parameters: view('uint32', 8)
  };
  try {
    const scan = options.expectationBased
      ? new GPUExpectationBasedSpaceTimeScan({
          ...common,
          maximumNeighbors: listLength,
          maximumReplicates: 1,
          maximumWindowLength: 1,
          output: {
            clusterZones: view('uint32', listLength),
            clusterStatistics: view('float32', 4),
            statistics: view('float32', 2),
            summary: view('uint32', 5)
          }
        })
      : new GPUSpatialScanStatistic({
          ...common,
          maximumWindowZones: listLength === zones ? zones : listLength - 1,
          maximumPermutations: 1,
          maximumClusters: 1,
          output: {
            clusterIndices: view('uint32', 4),
            clusterStatistics: view('float32', 8),
            statistics: view('float32', 2),
            summary: view('uint32', 4)
          }
        });
    expect(scan.estimatedIndexedCandidateSeparationCount).toBe(zones * zones);
    expect(scan.estimatedIndexedAxisSeparationCount).toBe(2 * zones * (zones + 1));
    expect(scan.estimatedIndexedSeparationCount).toBe(zones * zones + 2 * zones * (zones + 1));
    expect(scan.indexedRadixSortCount).toBe(2);
    const nodes = scan.getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    return {
      scan,
      nodeCount: nodes.length,
      sources,
      source: sources.find(kernel => kernel.id === 'scan-index-nearest')!.source,
      constants: sources.find(kernel => kernel.id === 'scan-index-nearest')!.constants
    };
  } finally {
    device.destroy();
  }
}

function getPositionKey(value: number): number {
  const bits = new Uint32Array(Float32Array.of(value === 0 ? 0 : value).buffer)[0];
  return ((bits & 0x80000000) !== 0 ? ~bits : bits ^ 0x80000000) >>> 0;
}

/** Execute production query WGSL with independently sorted inputs; headless tests exercise GPU sort wiring. */
function interpretNearest(
  source: string,
  positions: Float32Array,
  listLength: number,
  population?: Float32Array,
  countCandidates = false
) {
  if (countCandidates) {
    source = `var<private> candidateEvaluationCount: u32;
@group(0) @binding(8) var<storage, read_write> candidateCounts: array<u32>;
${source}`
      .replace(
        'fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {',
        'fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 { candidateEvaluationCount++;'
      )
      .replace('let row = index * LIST;', 'let row = index * LIST; candidateEvaluationCount = 0u;')
      .replace(/}\s*$/, 'candidateCounts[index] = candidateEvaluationCount; }');
  }
  const zones = positions.length / 2;
  const rows = Array.from({length: zones}, (_, row) => row);
  const order = (axis: number) =>
    [...rows].sort(
      (first, second) =>
        positions[first * 2 + axis] - positions[second * 2 + axis] || first - second
    );
  const rowsX = order(0);
  const rowsY = order(1);
  const memory: Record<string, Uint32Array | Float32Array> = {
    positions,
    neighbors: new Uint32Array(zones * listLength),
    distances: new Uint32Array(
      zones * (listLength + (population ? POISSON_BOUNDARY_WORDS_PER_ZONE : 0))
    ),
    ...(population ? {population} : {}),
    sortKeys: new Uint32Array(zones * listLength),
    sortZones: new Uint32Array(zones * listLength),
    sortedRowsX: Uint32Array.from(rowsX),
    sortedRowsY: Uint32Array.from(rowsY),
    sortedKeysX: Uint32Array.from(rowsX, row => getPositionKey(positions[row * 2])),
    sortedKeysY: Uint32Array.from(rowsY, row => getPositionKey(positions[row * 2 + 1])),
    candidateCounts: new Uint32Array(zones)
  };
  const bindings: Record<number, Record<number, Uint32Array | Float32Array>> = {0: {}};
  for (const binding of new WgslReflect(source).storage) {
    expect(memory[binding.name], binding.name).toBeDefined();
    bindings[binding.group][binding.binding] = memory[binding.name];
  }
  const workgroupSize = Number(source.match(/@workgroup_size\((\d+)\)/)![1]);
  const interpretedSource = `override QUERY_ROW: u32;\n${source}`
    .replace(/^\s*_ = [^;]+;/gm, '')
    .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
    .replace(
      `let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;`,
      'let index = QUERY_ROW;'
    );
  for (let query = 0; query < zones; query++)
    new WgslExec(new WgslParser().parse(interpretedSource)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      bindings,
      {
        constants: {
          QUERY_ROW: query,
          INVOCATION_COUNT: zones,
          WGSL_KERNEL_DISPATCH_X: 1,
          WGSL_KERNEL_DISPATCH_Y: 1
        }
      }
    );
  return {
    neighbors: Array.from(memory['neighbors']),
    candidateCounts: Array.from(memory['candidateCounts']),
    keys: Array.from(memory['distances']).slice(0, zones * listLength),
    boundaryPopulation: Array.from(
      new Float32Array(memory['distances'].buffer).slice(
        zones * listLength,
        zones * (listLength + 1)
      )
    ),
    boundaryCount: Array.from(
      memory['distances'].slice(zones * (listLength + 1), zones * (listLength + 2))
    )
  };
}

it('R5: a shrinking search radius discards old outer ties and retains the complete new boundary group', () => {
  // Zone 0 encounters the high-y pair first on the wider x axis. Its next circle
  // eventually contains only rows 0,3,4: population 1+8+16=25, not the old far pair.
  const positions = Float32Array.of(0, 0, -0.1, 10, 0.1, 10, -1, 0, 1, 0, 2, 0, 20, 0);
  const population = Float32Array.of(1, 2, 4, 8, 16, 32, 64);
  for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
    const {source} = createScene(7, 2, {nearestZoneSearch});
    const actual = interpretNearest(source, positions, 2, population);
    expect(actual.boundaryPopulation).toEqual([25, 6, 6, 9, 49, 48, 96]);
    expect(actual.boundaryCount).toEqual([3, 2, 2, 2, 3, 2, 2]);
  }
});

it('R12: boundary population rounds once from the exact positive float32 sum on either route', () => {
  const positions = Float32Array.of(-1, 0, 0, 0, 1, 0);
  // IEEE round-to-nearest of the exact sum, not a captured traversal result.
  for (const [population, expected] of [
    [Float32Array.of(0.2, 0.2, 1), 1.399999976158142],
    [Float32Array.of(1, 2 ** -24, 2 ** -25), 1.0000001192092896]
  ] as const) {
    for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
      const {source} = createScene(3, 2, {nearestZoneSearch});
      expect(interpretNearest(source, positions, 2, population).boundaryPopulation[1]).toBe(
        expected
      );
    }
  }
});

it('R12: exact boundary accumulator matches IEEE rounding independently of neighbor traversal', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  try {
    const view = createTransientView(graph, 'sum-input', 'float32', 3);
    const helpers = getScanBoundaryPopulationWGSL({view, integer: false, buckets: 1});
    const source = `@group(0) @binding(0) var<storage,read> population:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
const populationOffset:u32=0u;
${helpers}
@compute @workgroup_size(1) fn main(){resetBoundaryPopulation();addZoneBoundaryPopulation(0u);addZoneBoundaryPopulation(1u);addZoneBoundaryPopulation(2u);
for(var word=0u;word<10u;word++){result[word]=boundaryPopulationWords[word];} result[10]=bitcast<u32>(readBoundaryPopulation());}`;
    const result = new Uint32Array(11);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {
      0: {0: Float32Array.of(0.2, 0.2, 1), 1: result}
    });
    expect(Array.from(result.slice(0, 10))).toEqual([0, 0, 0, 0xd0000000, 0x2ccccc, 0, 0, 0, 0, 0]);
    expect(new Float32Array(result.buffer)[10]).toBe(1.399999976158142);
  } finally {
    device.destroy();
  }
});

for (const [name, coordinates, geographic] of [
  ['planar ties and duplicates', [0, 0, 1, 0, -1, 0, 0, 1, 0, -1, -0, 0, 20, 0], false],
  ['vertical extent chooses y', [2, 8, 2, -2, 2, 0, 2, 4, 2, 20, 2, -10, 2, 1], false],
  ['overflowing planar extent', [3e38, 0, -3e38, 0, 0, 0, 1e-20, 0, -1e-20, 0, 0, 1], false],
  [
    'great-circle seam and poles',
    [179.99, 0, -179.99, 0, 180, 0, -180, 0, 0, 90, 120, 90, 0, -90, 179.99, 0.01, 0, 0],
    true
  ],
  [
    'great-circle latitude boundary ties',
    [10, 60, 10, 60.01, 10, 59.99, 10.02, 60, 9.98, 60, 10, 60.02, 10, -60],
    true
  ],
  [
    'finite out-of-range latitudes fall back to unpruned discovery',
    [0, 100, 180, 100, 20, -100, 30, 90, 0, 0],
    true
  ]
] as const)
  it(`indexed keys and row ties equal exhaustive discovery: ${name}`, () => {
    const positions = Float32Array.from(coordinates);
    const zones = positions.length / 2;
    for (const listLength of [1, 3, zones]) {
      const indexed = interpretNearest(
        createScene(zones, listLength, {
          geographic,
          nearestZoneSearch: 'indexed',
          expectationBased: true
        }).source,
        positions,
        listLength
      );
      const exhaustive = interpretNearest(
        createScene(zones, listLength, {
          geographic,
          nearestZoneSearch: 'full-scan',
          expectationBased: true
        }).source,
        positions,
        listLength
      );
      expect(indexed.neighbors).toEqual(exhaustive.neighbors);
      expect(indexed.keys).toEqual(exhaustive.keys);
    }
  });

it('both scans choose indexed discovery at 65,536 zones, preserve lookahead, and expose full-scan work', () => {
  for (const expectationBased of [false, true]) {
    const {scan, nodeCount} = createScene(65536, 2, {expectationBased});
    expect(scan.nearestZoneSearch).toBe('indexed');
    expect(scan.estimatedFullScanSeparationCount).toBe(21474836480);
    if (scan instanceof GPUExpectationBasedSpaceTimeScan)
      expect(nodeCount).toBeLessThanOrEqual(scan.estimatedCommandNodeCount);
  }
  expect(getScanNearestZoneSearchPlan('scan', 100, 65).nearestZoneSearch).toBe('indexed');
  expect(getScanNearestZoneSearchPlan('scan', 100, 66).nearestZoneSearch).toBe('full-scan');
  expect(getScanNearestZoneSearchPlan('scan', 100, 2, 'full-scan').nearestZoneSearch).toBe(
    'full-scan'
  );
  expect(() => getScanNearestZoneSearchPlan('scan', 100, 66, 'indexed')).toThrow(
    /^scan nearestZoneSearch indexed listLength/
  );
  expect(() =>
    getScanNearestZoneSearchPlan('scan', 100, 2, 'bad' as GPUScanNearestZoneSearch)
  ).toThrow(/^scan nearestZoneSearch/);
});

it('largest indexed list and single-zone discovery stay within core shader limits', () => {
  for (const [zones, listLength] of [
    [66, 65],
    [1, 1]
  ]) {
    const {source, constants} = createScene(zones, listLength);
    expect(() => new WgslReflect(source)).not.toThrow();
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source, constants)).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
  }
  const singleton = interpretNearest(createScene(1, 1).source, Float32Array.of(-0, -2), 1);
  expect(singleton.neighbors).toEqual([0]);
  expect(singleton.keys).toEqual([0]);
});

it('auto uses one exhaustive discovery node when all 64 zones must be returned', () => {
  const scene = createScene(64, 64);
  expect(scene.scan.nearestZoneSearch).toBe('full-scan');
  expect(scene.sources.filter(kernel => kernel.id.includes('-nearest'))).toHaveLength(1);
});

it('spatial scan admits both discovery routes with a construction-time command-node bound', () => {
  for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
    const scene = createScene(65536, 2, {nearestZoneSearch});
    const estimate = (scene.scan as {estimatedCommandNodeCount?: number}).estimatedCommandNodeCount;
    expect(Number.isFinite(estimate)).toBe(true);
    expect(scene.nodeCount).toBeLessThanOrEqual(estimate!);
  }
});

it('coincident zero-distance ties displace higher-numbered centers in both routes', () => {
  for (const nearestZoneSearch of ['indexed', 'full-scan'] as const) {
    const result = interpretNearest(
      createScene(3, 1, {nearestZoneSearch, expectationBased: true}).source,
      new Float32Array(6),
      1
    );
    expect(result.neighbors).toEqual([0, 0, 0]);
  }
});

it('command estimates cover all models, parallel draw modes, batches and cluster rounds', () => {
  for (const model of ['poisson', 'bernoulli', 'space-time-permutation'] as const)
    for (const nearestZoneSearch of ['indexed', 'full-scan'] as const)
      for (const parallelDraws of [false, true]) {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device);
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
          format: Format,
          length: number
        ) => createTransientView(graph, `budget-${serial++}`, format, length);
        try {
          const scan = createSpatialScanStatisticForTesting({
            model,
            nearestZoneSearch,
            parallelDraws,
            positions: view('float32x2', 8),
            cases: view('uint32', 16),
            baseline: model === 'poisson' ? view('float32', 16) : undefined,
            controls: model === 'bernoulli' ? view('uint32', 16) : undefined,
            timeBuckets: 2,
            maximumPermutations: 7,
            maximumClusters: 3,
            maximumWindowZones: 2,
            batchReplicates: 2,
            parameters: view('uint32', 8),
            output: {
              clusterIndices: view('uint32', 12),
              clusterStatistics: view('float32', 24),
              statistics: view('float32', 8),
              summary: view('uint32', 4)
            }
          });
          expect(scan.getCommandNodes(graph).length).toBeLessThanOrEqual(
            scan.estimatedCommandNodeCount
          );
        } finally {
          device.destroy();
        }
      }
});

it('bounds indexed candidate and axis evaluations separately from the two radix sorts (PTS-06)', () => {
  // Each query visits each row at most once. At most N + 1 loop entries evaluate two bounds.
  // Equal latitudes make every latitude bound zero, so all N squared candidates can be visited.
  const plan = getScanNearestZoneSearchPlan('scan-budget', 100, 2);
  expect(plan.estimatedIndexedCandidateSeparationCount).toBe(10000);
  expect(plan.estimatedIndexedAxisSeparationCount).toBe(20200);
  expect(plan.estimatedIndexedSeparationCount).toBe(30200);
  expect(plan.indexedRadixSortCount).toBe(2);
});

it('equal-latitude geographic queries agree with independent great-circle enumeration (PTS-06)', () => {
  const positions = Float32Array.of(-170, 30, -80, 30, 0, 30, 40, 30, 120, 30, 179, 30);
  const listLength = 3;
  const actual = interpretNearest(
    createScene(6, listLength, {
      geographic: true,
      nearestZoneSearch: 'indexed',
      expectationBased: true
    }).source,
    positions,
    listLength,
    undefined,
    true
  );
  expect(actual.candidateCounts).toEqual([6, 6, 6, 6, 6, 6]);
  for (let center = 0; center < 6; center++) {
    // Spherical law of cosines; independent of the production haversine helper.
    const distances = Array.from({length: 6}, (_, row) => {
      const longitude = ((positions[row * 2] - positions[center * 2]) * Math.PI) / 180;
      return {row, angle: Math.acos(Math.min(1, 0.25 + 0.75 * Math.cos(longitude)))};
    }).sort((first, second) => first.angle - second.angle || first.row - second.row);
    expect(actual.neighbors.slice(center * listLength, (center + 1) * listLength)).toEqual(
      distances.slice(0, listLength).map(value => value.row)
    );
    for (let slot = 0; slot < listLength; slot++) {
      const key = new Float32Array(
        Uint32Array.of(actual.keys[center * listLength + slot]).buffer
      )[0];
      const angle = 2 * Math.asin(Math.sqrt(key));
      expect(Number.isFinite(angle)).toBe(true);
      expect(Math.abs(angle - distances[slot].angle)).toBeLessThan(2e-6);
    }
  }
});
