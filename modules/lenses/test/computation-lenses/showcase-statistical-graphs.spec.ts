// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GPUReduction} from '@luma.gl/graph/gpu-core';
import {GPU_PLANAR_SPATIAL_CONTEXT} from '@luma-contributors/analysis/analysis-core';
import {
  GPUSpatialScanStatistic,
  GPUMantelTest,
  GPUNeighborhoodSummary,
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  getGPUSpatialScanStatisticParameterValues,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  readFloat32,
  readUint32,
  submitGraph
} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {SpatialAnalysisResources} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/resources';
import {importGraphBuffer} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/graph-buffers';
import {addKernelPass} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/mode-kernels';
import {SCAN_MEMBERSHIP_BODY} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/time/space-time-clusters-membership';
import {getNeighborhoodSelectionBody} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/neighborhood-display';

it('explorer displays a valid mean beside median overflow and preserves no-data during input loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, 'explorer-median-loss');
  try {
    const graph = new GPUCommandGraph<void>(device);
    const create = (name: string, values: Float32Array | Uint32Array) =>
      resources.createBuffer(name, values);
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format
    ) => importGraphBuffer(graph, name, create(name, values), format, values.length);
    const loss = create('loss', Uint32Array.of(0));
    const source = create('source', Uint32Array.of(0));
    const selected = create('selected', new Float32Array(66));
    const extent = create('extent', new Float32Array(2));
    const medianOverflow = create('median-overflow', Uint32Array.of(0));
    const table = view('table', new Float32Array(132), 'float32');
    const modes = view('modes', new Uint32Array(66), 'uint32');
    const entropy = view('entropy', new Float32Array(66), 'float32');
    const selectedView = importGraphBuffer(graph, 'selected', selected, 'float32', 66);
    graph.add(
      new GPUNeighborhoodSummary({
        weights: {
          spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
          offsets: view(
            'offsets',
            Uint32Array.from({length: 67}, (_, row) => (row === 0 ? 0 : 65)),
            'uint32'
          ),
          neighbors: view(
            'neighbors',
            Uint32Array.from({length: 65}, (_, row) => row + 1),
            'uint32'
          ),
          weights: view('weights', new Float32Array(65).fill(1), 'float32'),
          status: {overflow: importGraphBuffer(graph, 'loss', loss, 'uint32', 1)}
        },
        values: view(
          'values',
          Float32Array.from({length: 66}, (_, row) => row),
          'float32'
        ),
        categories: view('categories', new Uint32Array(66), 'uint32'),
        statistics: ['mean', 'median'],
        maximumNeighbors: 64,
        includeFocal: false,
        output: {
          table,
          modes,
          entropy,
          status: {
            overflow: importGraphBuffer(graph, 'median-overflow', medianOverflow, 'uint32', 1)
          }
        }
      })
    );
    addKernelPass(graph, {
      id: 'neighborhood-select',
      invocationCount: 66,
      bindings: [
        {name: 'table', view: table, type: 'f32', access: 'read'},
        {name: 'modes', view: modes, type: 'u32', access: 'read'},
        {name: 'entropy', view: entropy, type: 'f32', access: 'read'},
        {
          name: 'source',
          view: importGraphBuffer(graph, 'source', source, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {name: 'selected', view: selectedView, type: 'f32', access: 'read_write'}
      ],
      body: getNeighborhoodSelectionBody(2)
    });
    graph.add(
      new GPUReduction({
        input: selectedView,
        output: importGraphBuffer(graph, 'extent', extent, 'float32', 2),
        operation: 'extent'
      })
    );
    const compiled = resources.track(await graph.compileAsync());
    submitGraph(device, compiled, undefined);
    expect(await readUint32(medianOverflow)).toEqual([1]);
    // (1+...+65)/65=33 is valid; only the median exceeds its per-row capacity of64.
    expect(await readFloat32(selected, 66)).toEqual([33, ...new Array(65).fill(NaN)]);
    expect(await readFloat32(extent, 2)).toEqual([33, 33]);
    source.write(Uint32Array.of(1));
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(selected, 66)).toEqual(new Array(66).fill(NaN));
    source.write(Uint32Array.of(0));
    loss.write(Uint32Array.of(1));
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(selected, 66)).toEqual(new Array(66).fill(NaN));
    loss.write(Uint32Array.of(0));
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(selected, 66)).toEqual([33, ...new Array(65).fill(NaN)]);
    expect(await readFloat32(extent, 2)).toEqual([33, 33]);
  } finally {
    resources.destroy();
  }
});

it('scene Mantel count port bounds attempted null draws including unusable permutations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, 'scene-mantel-prefix');
  try {
    const graph = new GPUCommandGraph<void>(device);
    const statistics = resources.createBuffer('statistics', 20 * 4);
    const attempted = resources.createBuffer('attempted', 4);
    const usable = resources.createBuffer('usable', 4);
    const parameters = getGPUSpaceTimeParameterValues({
      seed: 7,
      permutationCount: 19,
      timeThreshold: 1
    });
    graph.add(
      new GPUMantelTest({
        pairs: {
          spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
          offsets: importGraphBuffer(
            graph,
            'offsets',
            resources.createBuffer('offsets', Uint32Array.of(0, 2, 2, 2)),
            'uint32',
            4
          ),
          neighbors: importGraphBuffer(
            graph,
            'neighbors',
            resources.createBuffer('neighbors', Uint32Array.of(1, 2)),
            'uint32',
            2
          ),
          weights: importGraphBuffer(
            graph,
            'weights',
            resources.createBuffer('weights', Float32Array.of(1, 1)),
            'float32',
            2
          ),
          distances: importGraphBuffer(
            graph,
            'distances',
            resources.createBuffer('distances', Float32Array.of(1, 2)),
            'float32',
            2
          )
        },
        times: importGraphBuffer(
          graph,
          'times',
          resources.createBuffer('times', Float32Array.of(0, 0, 1)),
          'float32',
          3
        ),
        parameters: importGraphBuffer(
          graph,
          'parameters',
          resources.createBuffer('parameters', parameters),
          'uint32',
          parameters.length
        ),
        maximumPermutations: 19,
        output: {
          statistics: importGraphBuffer(graph, 'statistics', statistics, 'float32', 20),
          summary: importGraphBuffer(
            graph,
            'summary',
            resources.createBuffer('summary', GPU_SPACE_TIME_SUMMARY_LENGTH * 4),
            'float32',
            GPU_SPACE_TIME_SUMMARY_LENGTH
          ),
          permutationCount: importGraphBuffer(graph, 'attempted', attempted, 'uint32', 1),
          usablePermutationCount: importGraphBuffer(graph, 'usable', usable, 'uint32', 1)
        }
      })
    );
    submitGraph(device, resources.track(await graph.compileAsync()), undefined);
    const values = await readFloat32(statistics);
    const [attemptedCount] = await readUint32(attempted);
    const [usableCount] = await readUint32(usable);
    expect(attemptedCount).toBe(19);
    // Distances [1,2], observed gaps [0,1] => r=1. If the unique time is moved
    // to the focal event, gaps [1,1] have zero variance and that null draw is unusable.
    expect(Number.isFinite(values[0])).toBe(true);
    expect(values[0]).toBeCloseTo(1, 6);
    const finite = values.slice(1, attemptedCount + 1).filter(Number.isFinite);
    expect(finite).toHaveLength(usableCount);
    expect(usableCount).toBeGreaterThan(0);
    expect(usableCount).toBeLessThan(attemptedCount);
    finite.forEach(value => expect(Math.abs(value)).toBeCloseTo(1, 6));
  } finally {
    resources.destroy();
  }
});

it('scene scan status distinguishes invalid exposure from valid empty input and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources = new SpatialAnalysisResources(device, 'scene-scan-status');
  try {
    const graph = new GPUCommandGraph<void>(device);
    const cases = resources.createBuffer('cases', Uint32Array.of(1, 0, 0, 0));
    const baseline = resources.createBuffer('baseline', Float32Array.of(0, 1, 1, 1));
    const invalid = resources.createBuffer('invalid', 4);
    const indices = resources.createBuffer('indices', 16);
    const membership = resources.createBuffer('membership', 12);
    const positions = resources.createBuffer('positions', Float32Array.of(0, 0, 10, 0));
    const summary = resources.createBuffer(
      'summary',
      GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH * 4
    );
    const statistics = resources.createBuffer('cluster-statistics', 8 * 4);
    const clusterIndicesView = importGraphBuffer(graph, 'indices', indices, 'uint32', 4);
    const clusterStatisticsView = importGraphBuffer(
      graph,
      'cluster-statistics',
      statistics,
      'float32',
      8
    );
    const positionsView = importGraphBuffer(graph, 'positions', positions, 'float32x2', 2);
    const invalidCountView = importGraphBuffer(graph, 'invalid', invalid, 'uint32', 1);
    const parameters = getGPUSpatialScanStatisticParameterValues({
      seed: 7,
      permutationCount: 1,
      maximumPopulationFraction: 0.5,
      maximumWindowZones: 1,
      maximumTimeBuckets: 1
    });
    graph.add(
      new GPUSpatialScanStatistic({
        spatialContext: GPU_PLANAR_SPATIAL_CONTEXT,
        positions: positionsView,
        cases: importGraphBuffer(graph, 'cases', cases, 'uint32', 4),
        baseline: importGraphBuffer(graph, 'baseline', baseline, 'float32', 4),
        timeBuckets: 2,
        maximumClusters: 1,
        maximumPermutations: 1,
        parameters: importGraphBuffer(
          graph,
          'parameters',
          resources.createBuffer('parameters', parameters),
          'uint32',
          parameters.length
        ),
        output: {
          clusterIndices: clusterIndicesView,
          clusterStatistics: clusterStatisticsView,
          statistics: importGraphBuffer(
            graph,
            'statistics',
            resources.createBuffer('statistics', 8),
            'float32',
            2
          ),
          summary: importGraphBuffer(
            graph,
            'summary',
            summary,
            'uint32',
            GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
          ),
          status: {invalidCount: invalidCountView}
        }
      })
    );
    addKernelPass(graph, {
      id: 'scene-membership',
      invocationCount: 3,
      declarations: 'const ZONES: u32 = 2u; const CLUSTERS: u32 = 1u;',
      bindings: [
        {name: 'indices', view: clusterIndicesView, type: 'u32', access: 'read'},
        {name: 'stats', view: clusterStatisticsView, type: 'f32', access: 'read'},
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'invalidCount', view: invalidCountView, type: 'u32', access: 'read'},
        {
          name: 'alpha',
          view: importGraphBuffer(
            graph,
            'alpha',
            resources.createBuffer('alpha', Float32Array.of(1)),
            'float32',
            1
          ),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'membership',
          view: importGraphBuffer(graph, 'membership', membership, 'uint32', 3),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: SCAN_MEMBERSHIP_BODY
    });
    const compiled = resources.track(await graph.compileAsync());
    submitGraph(device, compiled, undefined);
    expect(await readUint32(invalid)).toEqual([1]);
    expect(await readUint32(membership)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    cases.write(Uint32Array.of(0, 0, 0, 0));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(invalid)).toEqual([0]);
    expect((await readUint32(summary))[0]).toBe(0);
    expect(await readUint32(membership)).toEqual([0, 0, 0xffffffff]);
    cases.write(Uint32Array.of(3, 0, 0, 1));
    baseline.write(Float32Array.of(1, 1, 1, 1));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(invalid)).toEqual([0]);
    expect((await readUint32(summary))[0]).toBe(1);
    expect(await readUint32(membership)).toEqual([1, 0, 0xffffffff]);
    const values = await readFloat32(statistics);
    expect(Number.isFinite(values[0])).toBe(true);
    // Conditional Poisson LLR: 3*log(3/1) + 1*log(1/3) = 2*log(3).
    expect(values[0]).toBeCloseTo(2.1972245773362196, 5);
  } finally {
    resources.destroy();
  }
});
