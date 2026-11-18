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
      distances: new Uint32Array(zones * 2),
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
function interpretNearest(source: string, positions: Float32Array, listLength: number) {
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
    distances: new Uint32Array(zones * listLength),
    sortKeys: new Uint32Array(zones * listLength),
    sortZones: new Uint32Array(zones * listLength),
    sortedRowsX: Uint32Array.from(rowsX),
    sortedRowsY: Uint32Array.from(rowsY),
    sortedKeysX: Uint32Array.from(rowsX, row => getPositionKey(positions[row * 2])),
    sortedKeysY: Uint32Array.from(rowsY, row => getPositionKey(positions[row * 2 + 1]))
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
  return {neighbors: Array.from(memory['neighbors']), keys: Array.from(memory['distances'])};
}

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
