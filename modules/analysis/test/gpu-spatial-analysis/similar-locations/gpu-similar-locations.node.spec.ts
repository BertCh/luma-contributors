// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  getGPUSimilarLocationsMaximumValueCount,
  GPUSimilarLocations
} from '../../../src/gpu-spatial-analysis/similar-locations/index';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

it('checks numerical helper bounds and finite float32 attribute weights', () => {
  expect(getGPUSimilarLocationsParameterLength(3)).toBe(7);
  for (const columnCount of [0, -1, 1.5, NaN])
    expect(() => getGPUSimilarLocationsParameterLength(columnCount)).toThrow();
  expect(() =>
    getGPUSimilarLocationsParameterValues({resultCount: 3, weights: [1e300]}, 1)
  ).toThrow();
  expect(() => getGPUSimilarLocationsParameterValues({resultCount: 2 ** 24 + 1}, 1)).toThrow();
  expect(
    Array.from(getGPUSimilarLocationsParameterValues({resultCount: 3, weights: [2, 0.5, 0]}, 3))
  ).toEqual([3, 0, 1, 0, 2, 0.5, 0]);
  const device = createNullWebGPUDevice();
  try {
    Object.assign(device.limits, {maxStorageBufferBindingSize: 128 * 2 ** 20});
    expect(getGPUSimilarLocationsMaximumValueCount(device)).toBe(33554432);
  } finally {
    device.destroy();
  }
});

// At and above 4096 rows `'rank'` sorts one column at a time.
const ROWS = 4096;
const RESULT_COUNT = 8;

function build(
  attributeCount: number,
  standardization: 'rank' | 'z-score' = 'rank',
  oversized = false,
  rowCount = ROWS
) {
  return buildGraph(attributeCount, standardization, oversized, rowCount).nodes;
}

/** Builds the graph and records every transient buffer the contributor allocates. */
function buildGraph(
  attributeCount: number,
  standardization: 'rank' | 'z-score' = 'rank',
  oversized = false,
  rowCount = ROWS
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const transientIds: string[] = [];
  const createTransientBuffer = graph.createTransientBuffer.bind(graph);
  graph.createTransientBuffer = (props: Parameters<typeof createTransientBuffer>[0]) => {
    transientIds.push(String(props.id));
    return createTransientBuffer(props);
  };
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `similar-${serial++}`, format, length);
  const nodes = new GPUSimilarLocations({
    id: 'similar',
    values: view('float32', rowCount * attributeCount),
    columnCount: attributeCount,
    referenceMask: view('uint32', rowCount),
    parameters: view('float32', getGPUSimilarLocationsParameterLength(attributeCount)),
    standardization,
    resultCapacity: RESULT_COUNT,
    output: {
      ranks: view('uint32', rowCount),
      distances: view('float32', rowCount),
      topIds: view('uint32', RESULT_COUNT + (oversized ? 3 : 0)),
      status: {count: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
  return {graph, nodes, transientIds};
}

it('GPUSimilarLocations relative zscore and oversized topIds shaders parse as WGSL', () => {
  for (const {id, source} of getGPUCommandNodeKernelSources(build(1, 'z-score', true)))
    expect(() => new WgslReflect(source), id).not.toThrow();
});

it('W5-19 omits unused z-score materialization and retains both rank materializations', () => {
  for (const standardization of ['z-score'] as const) {
    const kernels = getGPUCommandNodeKernelSources(build(1, standardization));
    expect(kernels.some(kernel => kernel.id.includes('standardize-zscore'))).toBe(false);
    expect(kernels.some(kernel => /var<storage[^>]*> standardized:/.test(kernel.source))).toBe(
      false
    );
  }
  for (const rowCount of [8, 4096]) {
    const kernels = getGPUCommandNodeKernelSources(build(1, 'rank', false, rowCount));
    expect(
      kernels.some(kernel => /var<storage, read_write> standardized:/.test(kernel.source))
    ).toBe(true);
    expect(kernels.some(kernel => /var<storage, read> standardized:/.test(kernel.source))).toBe(
      true
    );
  }
});

it('W5-19 allocates no standardized matrix on z-score and wires the rank writer to its readers', () => {
  for (const standardization of ['z-score'] as const) {
    const {transientIds} = buildGraph(2, standardization);
    expect(transientIds).not.toContain('similar-standardized');
    expect(transientIds.some(id => id.includes('standardized'))).toBe(false);
  }
  // 8 rows take the direct rank kernel; 4201 rows take the per-column sorted route.
  for (const [rowCount, writerIds] of [
    [8, ['similar-standardize-rank']],
    [4201, ['similar-rank-assign-0', 'similar-rank-assign-1']]
  ] as const) {
    const {graph, nodes, transientIds} = buildGraph(2, 'rank', false, rowCount);
    expect(transientIds.filter(id => id === 'similar-standardized')).toHaveLength(1);
    const nodeIds = nodes.map(node => node.id);
    const getBinding = (nodeId: string, name: string) => {
      const record = getWGSLKernelNodeRecord(graph, nodeId);
      expect(record, nodeId).toBeDefined();
      const binding = record!.bindings.find(entry => entry.name === name);
      expect(binding, `${nodeId} ${name}`).toBeDefined();
      return binding!;
    };
    const written = writerIds.map(writerId => {
      expect(nodeIds, writerId).toContain(writerId);
      const binding = getBinding(writerId, 'standardized');
      expect(binding.access).toBe('read_write');
      return binding.view;
    });
    for (const view of written) expect(view).toBe(written[0]);
    for (const readerId of ['similar-reference', 'similar-distance']) {
      const binding = getBinding(readerId, 'standardized');
      expect(binding.access).toBe('read');
      expect(binding.view).toBe(written[0]);
    }
  }
});

it('GPUSimilarLocations rank columns share one pipeline per kernel (rule 12) and parse as WGSL', () => {
  expect(getGPUCommandNodePipelineKeys(build(4)).size).toBe(
    getGPUCommandNodePipelineKeys(build(3)).size
  );
  for (const {id, source} of getGPUCommandNodeKernelSources(build(3))) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
});

it('rejects an untyped unknown ranking direction', () => {
  expect(() =>
    getGPUSimilarLocationsParameterValues({resultCount: 2, direction: 'least-similar'} as never, 1)
  ).toThrow(/direction.*most.*least/);
});

it('accepts both population z-score spellings and diagnoses unknown standardization', () => {
  expect(() => build(1, 'z-score')).not.toThrow();
  expect(() => build(1, 'zscore' as never)).toThrow(/standardization.*z-score/);
  expect(() => build(1, 'z_score' as never)).toThrow(/standardization.*z-score.*rank.*z_score/);
});

it('preflights rank reference storage without a z-score moments allocation', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.assign(device.limits, {maxStorageBufferBindingSize: 128 * 2 ** 20});
    const graph = new GPUCommandGraph(device);
    const columnCount = 9_000_000;
    const solver = new GPUSimilarLocations({
      values: createTransientView(graph, 'values', 'float32', 2 * columnCount),
      columnCount,
      referenceMask: createTransientView(graph, 'mask', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', columnCount + 4),
      standardization: 'rank',
      resultCapacity: 2,
      output: {
        ranks: createTransientView(graph, 'ranks', 'uint32', 2),
        distances: createTransientView(graph, 'distances', 'float32', 2),
        topIds: createTransientView(graph, 'ids', 'uint32', 2),
        status: {count: createTransientView(graph, 'count', 'uint32', 1)}
      }
    });
    // Largest allocation is the 72 MB two-row matrix, below the 128 MiB binding limit.
    expect(() => solver.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
