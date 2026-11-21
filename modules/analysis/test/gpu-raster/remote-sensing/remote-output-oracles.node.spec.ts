// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {computeRasterRegionGraphOnCPU} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {makeCPUInputRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {CPURasterRegionGraphOperation} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {GPURasterRegionGraphOperation} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {snapshotRasterRegionGraphOperationProps} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {GPUCommandGraph, createTransientView, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURasterRegionGraph} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph';
import {GPURasterConservativeRemapping} from '../../../src/raster-analysis/remote-sensing/gpu-raster-conservative-remapping';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

it('region CPU twin honors a one-round adaptive budget and reports the remaining eligible merge', () => {
  const result = computeRasterRegionGraphOnCPU(
    new Uint32Array([0, 1, 2]),
    new Float32Array([0, 2, 4]),
    {
      grid: makeGPURasterMetadata({width: 3, height: 1}),
      regionCount: 3,
      mergeMode: 'adaptive',
      mergeThreshold: 3,
      maximumIterations: 1
    }
  );
  // Equal weight-2 edges tie: (0,1) wins. Its mean becomes 1; edge to region 2 has weight 3.
  expect({
    mapping: Array.from(result.mapping),
    pending: Array.from(result.iterationLimitReached)
  }).toEqual({mapping: [0, 0, 2], pending: [1]});
  expectLiteralRows(result.adjacency, [0, 1, 0, 1, 0, 1, 0, 1, 0], 0);
  expectLiteralRows(result.weights, [NaN, NaN, NaN, NaN, NaN, 3, NaN, 3, NaN], 0);
  expectLiteralRows(result.mapping, [0, 0, 2], 0);
  expectLiteralRows(result.iterationLimitReached, [1], 0);
});

it('fractional remap prepare shader writes source rows distinct from candidate slots and nonbinary areas', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const indices = createTransientView(graph, 'indices', 'uint32', 6);
    const areas = createTransientView(graph, 'areas', 'float32', 6);
    const contributor = new GPURasterConservativeRemapping({
      id: 'fractional-remap',
      source: {width: 4, height: 1, origin: [0, 0], spacing: [1, 1]},
      destination: {width: 1, height: 1, origin: [1.25, 0], spacing: [1.25, 1]},
      values: createTransientView(graph, 'values', 'float32', 4),
      preparedWeights: {indices, areas},
      normalization: 'destination-area',
      output: {
        values: createTransientView(graph, 'result', 'float32', 1),
        coveredArea: createTransientView(graph, 'covered', 'float32', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    });
    const memory = createCPUGraphMemory();
    memory.write(indices, new Uint32Array(6).fill(123));
    memory.write(areas, new Float32Array(6).fill(123));
    executeCommandNodesOnCPU(
      [contributor.getCommandNodes(graph).find(node => node.id === 'fractional-remap-prepare')!],
      memory
    );
    // Shapely rectangle intersections, remote-output-oracles.oracle.py.
    expectLiteralRows(
      memory.readUint32(indices),
      [1, 2, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
      0
    );
    expectLiteralRows(memory.readFloat32(areas), [0.75, 0.5, 0, 0, 0, 0], 0);
  } finally {
    device.destroy();
  }
});

it('constructed CPU and GPU region operations retain route, grid, options and nested port names', () => {
  for (const location of ['cpu', 'gpu'] as const) {
    const props = {
      id: 'captured',
      grid: {
        ...makeGPURasterMetadata({width: 4, height: 1}),
        levelZeroOrigin: [3, 5] as [number, number],
        coordinateReferenceSystem: {authority: 'EPSG:3857', projectionJson: {axis: {order: 'east'}}}
      },
      regionCount: 4,
      labels: 'labels',
      values: 'values',
      mergeMode: 'fixed' as const,
      searchStructure: 'dense' as 'dense' | 'boruvka',
      mergeThreshold: 3,
      maximumIterations: 1,
      output: {
        adjacency: 'adjacency',
        weights: 'weights',
        mapping: 'originalMapping',
        status: {iterationLimitReached: 'originalPending'}
      }
    };
    const operation =
      location === 'cpu'
        ? new CPURasterRegionGraphOperation(props)
        : new GPURasterRegionGraphOperation(props);
    const execute = () => {
      if (operation instanceof CPURasterRegionGraphOperation) {
        const rows = operation.execute({
          inputs: {
            labels: makeCPUInputRows('uint32', new Uint32Array([0, 1, 2, 3])),
            values: makeCPUInputRows('float32', new Float32Array([0, 1, 4, 5]))
          }
        }) as Record<string, Float32Array | Uint32Array>;
        expect(Object.keys(rows)).toEqual([
          'adjacency',
          'weights',
          'originalMapping',
          'originalPending'
        ]);
        expectLiteralRows(rows['originalMapping'], [0, 0, 1, 2], 0);
        expectLiteralRows(rows['originalPending'], [1], 0);
      } else {
        const device = createNullWebGPUDevice();
        try {
          const graph = new GPUCommandGraph(device);
          const views = {
            labels: createTransientView(graph, 'labels', 'uint32', 4),
            values: createTransientView(graph, 'values', 'float32', 4),
            ...Object.fromEntries(
              Object.entries(operation.outputs).map(([name, output]) => [
                name,
                createTransientView(graph, name, output.format, output.length)
              ])
            )
          };
          const contributor = operation.createContributor(
            port => views[port as keyof typeof views] as never
          ) as GPURasterRegionGraph;
          expect(contributor.props.searchStructure).toBe('dense');
          expect(contributor.props.maximumIterations).toBe(1);
          expect(contributor.props.grid.width).toBe(4);
          expect(contributor.props.grid.affine).toEqual([1, 0, 0, 0, -1, 0]);
          expect(contributor.props.grid.levelZeroOrigin).toEqual([3, 5]);
          expect(contributor.props.grid.coordinateReferenceSystem).toEqual({
            authority: 'EPSG:3857',
            projectionJson: {axis: {order: 'east'}}
          });
        } finally {
          device.destroy();
        }
      }
    };
    execute();
    props.searchStructure = 'boruvka';
    execute();
    props.maximumIterations = 2;
    props.mergeThreshold = 0;
    props.grid.width = 1;
    (props.grid.affine as unknown as number[])[0] = 100;
    props.grid.levelZeroOrigin[0] = 99;
    props.grid.coordinateReferenceSystem.authority = 'changed';
    props.grid.coordinateReferenceSystem.projectionJson.axis.order = 'changed';
    props.output.mapping = 'changedMapping';
    props.output.status.iterationLimitReached = 'changedPending';
    execute();
  }
});

it('direct region contributor retains its one-round configuration and bound output views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const mapping = createTransientView(graph, 'original-mapping', 'uint32', 4);
    const pending = createTransientView(graph, 'original-pending', 'uint32', 1);
    const props = {
      id: 'direct-captured',
      grid: makeGPURasterMetadata({width: 4, height: 1}),
      labels: createTransientView(graph, 'labels', 'uint32', 4),
      values: createTransientView(graph, 'values', 'float32', 4),
      regionCount: 4,
      searchStructure: 'dense' as 'dense' | 'boruvka',
      mergeThreshold: 3,
      maximumIterations: 1,
      output: {
        adjacency: createTransientView(graph, 'adjacency', 'uint32', 16),
        weights: createTransientView(graph, 'weights', 'float32', 16),
        mapping,
        status: {iterationLimitReached: pending}
      }
    };
    const contributor = new GPURasterRegionGraph(props);
    props.maximumIterations = 2;
    const nodes = contributor.getCommandNodes(graph);
    const merges = nodes.filter(node => node.id.startsWith('direct-captured-merge-'));
    const memory = createCPUGraphMemory();
    // Hand-counted chain edges with weights [1,3,1]; one propagation yields [0,0,1,2].
    memory.write(
      props.output.weights,
      new Float32Array([NaN, 1, NaN, NaN, 1, NaN, 3, NaN, NaN, 3, NaN, 1, NaN, NaN, 1, NaN])
    );
    const source = merges[0].resources!.flatMap(resource =>
      'buffer' in resource &&
      resource.buffer instanceof GraphDataView &&
      resource.buffer.buffer.id === 'direct-captured-mapping'
        ? [resource.buffer]
        : []
    )[0];
    memory.write(source, new Uint32Array([0, 1, 2, 3]));
    executeCommandNodesOnCPU(
      [...merges, nodes.find(node => node.id === 'direct-captured-finish')!],
      memory
    );
    expect(Array.from(memory.readUint32(mapping))).toEqual([0, 0, 1, 2]);
    expectLiteralRows(memory.readUint32(pending), [1], 0);
    expect(merges).toHaveLength(1);
    props.searchStructure = 'boruvka';
    props.grid.width = 1;
    props.mergeThreshold = 0;
    props.output.mapping = createTransientView(graph, 'changed-mapping', 'uint32', 4);
    props.output.status.iterationLimitReached = createTransientView(
      graph,
      'changed-pending',
      'uint32',
      1
    );
    expect(contributor.props.maximumIterations).toBe(1);
    expect(contributor.props.grid.width).toBe(4);
    expect(contributor.props.mergeThreshold).toBe(3);
    expect(contributor.props.output.mapping).toBe(mapping);
    expect(contributor.props.output.status.iterationLimitReached).toBe(pending);
  } finally {
    device.destroy();
  }
});

it.each([
  'cpu',
  'gpu'
] as const)('region %s wrapper accepts and snapshots proxy CRS descriptors', location => {
  const projection = {name: 'original', axes: [{direction: 'east'}]};
  const backing = {
    authority: 'EPSG:3857',
    wellKnownText: 'original-wkt',
    projectionJson: new Proxy(projection, {})
  };
  const props = {
    id: 'proxy-crs',
    grid: {
      ...makeGPURasterMetadata({width: 4, height: 1}),
      coordinateReferenceSystem: new Proxy(backing, {})
    },
    labels: 'labels',
    values: 'values',
    regionCount: 4,
    searchStructure: 'dense' as const,
    maximumIterations: 1,
    mergeThreshold: 3,
    output: {
      adjacency: 'adjacency',
      weights: 'weights',
      mapping: 'mapping',
      status: {iterationLimitReached: 'pending'}
    }
  };
  const operation =
    location === 'cpu'
      ? new CPURasterRegionGraphOperation(props)
      : new GPURasterRegionGraphOperation(props);
  // The CPU wrapper and GPU wrapper use this same snapshot function; GPU lowering also
  // exposes the captured metadata on its direct contributor for an identity-independent check.
  const captured = snapshotRasterRegionGraphOperationProps(props);
  backing.authority = 'changed';
  backing.wellKnownText = 'changed';
  projection.name = 'changed';
  projection.axes[0].direction = 'changed';
  expect(captured.grid.coordinateReferenceSystem).toEqual({
    authority: 'EPSG:3857',
    wellKnownText: 'original-wkt',
    projectionJson: {name: 'original', axes: [{direction: 'east'}]}
  });
  if (operation instanceof CPURasterRegionGraphOperation) {
    const rows = operation.execute({
      inputs: {
        labels: makeCPUInputRows('uint32', new Uint32Array([0, 1, 2, 3])),
        values: makeCPUInputRows('float32', new Float32Array([0, 1, 4, 5]))
      }
    }) as Record<string, Uint32Array>;
    expectLiteralRows(rows['mapping'], [0, 0, 1, 2], 0);
  } else {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const views = {
        labels: createTransientView(graph, 'labels', 'uint32', 4),
        values: createTransientView(graph, 'values', 'float32', 4),
        ...Object.fromEntries(
          Object.entries(operation.outputs).map(([name, output]) => [
            name,
            createTransientView(graph, name, output.format, output.length)
          ])
        )
      };
      const contributor = operation.createContributor(
        port => views[port as keyof typeof views] as never
      ) as GPURasterRegionGraph;
      expect(contributor.props.grid.coordinateReferenceSystem).toEqual(
        captured.grid.coordinateReferenceSystem
      );
    } finally {
      device.destroy();
    }
  }
});

it('fixed CPU regions distinguish a dense round from one Boruvka phase and expose unfinished mappings', () => {
  for (const [searchStructure, maximumIterations, expectedMapping, pending] of [
    ['auto', 1, [0, 0, 1, 2], 1],
    ['dense', 2, [0, 0, 0, 1], 1],
    ['boruvka', 1, [0, 0, 2, 2], 1],
    ['boruvka', 2, [0, 0, 0, 0], 0]
  ] as const) {
    // Line 0--1--2--3 with weights 1,3,1: one Boruvka phase selects the two weight-1 edges.
    const result = computeRasterRegionGraphOnCPU(
      new Uint32Array([0, 1, 2, 3]),
      new Float32Array([0, 1, 4, 5]),
      {
        grid: makeGPURasterMetadata({width: 4, height: 1}),
        regionCount: 4,
        mergeMode: 'fixed',
        searchStructure,
        mergeThreshold: 3,
        maximumIterations
      }
    );
    expectLiteralRows(result.adjacency, [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], 0);
    expectLiteralRows(
      result.weights,
      [NaN, 1, NaN, NaN, 1, NaN, 3, NaN, NaN, 3, NaN, 1, NaN, NaN, 1, NaN],
      0
    );
    expectLiteralRows(result.mapping, expectedMapping, 0);
    expectLiteralRows(result.iterationLimitReached, [pending], 0);
  }
});

it('checkerboard routes retain at most one distinct boundary pair', () => {
  const size = 768;
  const labels = Uint32Array.from(
    {length: size * size},
    (_, cell) => ((cell % size) + Math.floor(cell / size)) % 2
  );
  const values = Float32Array.from(labels);
  for (const [mergeMode, searchStructure] of [
    ['adaptive', 'dense'],
    ['fixed', 'dense'],
    ['fixed', 'boruvka']
  ] as const) {
    let retainedBoundaryObjects = 0;
    const push = Array.prototype.push;
    Array.prototype.push = function (...items: unknown[]) {
      retainedBoundaryObjects += items.filter(
        item =>
          item &&
          typeof item === 'object' &&
          Object.keys(item).length === 2 &&
          'first' in item &&
          'second' in item
      ).length;
      return push.apply(this, items);
    };
    const before = process.memoryUsage().rss;
    let result: ReturnType<typeof computeRasterRegionGraphOnCPU>;
    try {
      result = computeRasterRegionGraphOnCPU(labels, values, {
        grid: makeGPURasterMetadata({width: size, height: size}),
        regionCount: 2,
        mergeMode,
        searchStructure,
        mergeThreshold: 1,
        maximumIterations: 1
      });
    } finally {
      Array.prototype.push = push;
    }
    console.info(
      JSON.stringify({
        review: 'R3',
        mergeMode,
        searchStructure,
        retainedBoundaryObjects,
        rssDelta: process.memoryUsage().rss - before
      })
    );
    expectLiteralRows(result!.mapping, [0, 0], 0);
    expectLiteralRows(result!.iterationLimitReached, [0], 0);
    expect(retainedBoundaryObjects).toBe(searchStructure === 'boruvka' ? 1 : 0);
  }
});
