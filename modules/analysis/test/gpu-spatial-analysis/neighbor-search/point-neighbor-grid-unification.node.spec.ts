// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH,
  type GPUNeighborSearchProps
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 32;
const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNeighborSearchProps> = {}
): GPUNeighborSearchProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    mode: 'knn',
    k: 4,
    positions: view('float32x2', ROWS),
    times: view('float32', ROWS),
    temporalContext: CONTEXT,
    parameters: view('float32', GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH),
    gridSize: [4, 4],
    output: {
      offsets: view('uint32', ROWS + 1),
      neighbors: view('uint32', ROWS * 4),
      weights: view('float32', ROWS * 4),
      status: {overflow: view('uint32', 1)}
    },
    ...overrides
  };
}

function createGraph(): GPUCommandGraph {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: 'point-neighbor-grid-unification'});
}

it('accepts the time axis in both modes and both join kinds', () => {
  const graph = createGraph();
  expect(() => new GPUNeighborSearch(createProps(graph))).not.toThrow();
  expect(
    () => new GPUNeighborSearch(createProps(graph, {mode: 'radius', timeBucketCount: 8}))
  ).not.toThrow();
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          queryPositions: createTransientView(graph, 'query', 'float32x2', ROWS),
          queryTimes: createTransientView(graph, 'query-times', 'float32', ROWS)
        })
      )
  ).not.toThrow();
});

it('names the rule when the time inputs are inconsistent', () => {
  const graph = createGraph();
  const create = (overrides: Partial<GPUNeighborSearchProps>) => () =>
    new GPUNeighborSearch(createProps(graph, overrides));
  expect(create({temporalContext: undefined})).toThrow(/times need a temporalContext/);
  expect(create({searchStructure: 'bvh'})).toThrow(/times need searchStructure/);
  expect(create({times: createTransientView(graph, 'short', 'float32', ROWS - 1)})).toThrow(
    /times length must equal positions length/
  );
  expect(create({queryPositions: createTransientView(graph, 'q', 'float32x2', ROWS)})).toThrow(
    /queryTimes are required/
  );
  expect(create({queryTimes: createTransientView(graph, 'q2', 'float32', ROWS)})).toThrow(
    /queryTimes need queryPositions/
  );
  expect(create({times: undefined})).toThrow(/temporalContext, queryTimes and timeBucketCount/);
  expect(create({times: undefined, temporalContext: undefined, timeBucketCount: 4})).toThrow(
    /temporalContext, queryTimes and timeBucketCount/
  );
  expect(
    create({
      parameters: createTransientView(graph, 'p', 'float32', GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH)
    })
  ).toThrow(/parameters must hold 15/);
  for (const timeBucketCount of [0, 1.5, GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS + 1]) {
    expect(create({timeBucketCount}), `buckets ${timeBucketCount}`).toThrow(/timeBucketCount/);
  }
  expect(create({temporalContext: {...CONTEXT, encoding: 'int64-words'}})).toThrow();
});

it('checks the bucketed cell count against the grid index limit', () => {
  const graph = createGraph();
  const search = new GPUNeighborSearch(
    createProps(graph, {gridSize: [8192, 8192], timeBucketCount: 4096})
  );
  expect(() => search.getCommandNodes(graph)).toThrow(/gridSize cells must be at most/);
});

it('packs the temporal words only for a time window, and needs both of them', () => {
  const base = {bounds: [0, 0, 1, 1] as [number, number, number, number]};
  expect(getGPUNeighborSearchParameterValues(base).length).toBe(
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const values = getGPUNeighborSearchParameterValues({
    ...base,
    timeRadius: 6,
    timeBounds: [-2, 40]
  });
  expect(values.length).toBe(GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH);
  expect(Array.from(values.slice(12))).toEqual([6, -2, 40]);
  expect(() => getGPUNeighborSearchParameterValues({...base, timeRadius: 6})).toThrow(/together/);
  expect(() => getGPUNeighborSearchParameterValues({...base, timeBounds: [0, 1]})).toThrow(
    /together/
  );
});

it('stays inside the WGSL private memory limit at the largest k', () => {
  const graph = createGraph();
  const k = GPU_NEIGHBOR_SEARCH_MAXIMUM_K;
  const nodes = new GPUNeighborSearch(
    createProps(graph, {
      k,
      output: {
        offsets: createTransientView(graph, 'o', 'uint32', ROWS + 1),
        neighbors: createTransientView(graph, 'n', 'uint32', ROWS * k),
        weights: createTransientView(graph, 'w', 'float32', ROWS * k),
        status: {overflow: createTransientView(graph, 'v', 'uint32', 1)}
      }
    })
  ).getCommandNodes(graph);
  const recordedKernelSources1 = getParsedKernelSources(nodes);
  for (const {id: kernelId, source: source, constants} of recordedKernelSources1.configurations) {
    expect(getPrivateMemoryBytes(source, constants), kernelId).toBeLessThanOrEqual(8192);
  }
  expect(getGPUCommandNodePipelineKeys(nodes).size).toBeGreaterThan(0);
});

it('reports the k-nearest list against the storage binding at the largest k', () => {
  const graph = createGraph();
  const rows = 2 ** 24;
  const k = GPU_NEIGHBOR_SEARCH_MAXIMUM_K;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `large-${serial++}`, format, length);
  const search = new GPUNeighborSearch(
    createProps(graph, {
      k,
      positions: view('float32x2', rows),
      times: view('float32', rows),
      output: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', 1024),
        weights: view('float32', 1024),
        status: {overflow: view('uint32', 1)}
      }
    })
  );
  expect(() => search.getCommandNodes(graph)).toThrow(/k-nearest lists/);
});
