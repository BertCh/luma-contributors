// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUChunkedSpatialWeightsTranspose,
  type GPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeightsTransposeOutput
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {GPU_GREAT_CIRCLE_SPATIAL_CONTEXT} from '../../../src/analysis-core/contracts/spatial-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createSetup(inputRows = [2, 4], inputCapacity = 8) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const createSpatialWeights = (rows: number, capacity: number) => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: view('float32', capacity)
  });
  let rowStart = 0;
  const weights: GPUChunkedSpatialWeights = {
    rowCount: inputRows.reduce((sum, count) => sum + count, 0),
    spatialContext: GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
    information: {
      rowCount: inputRows.reduce((sum, count) => sum + count, 0),
      transform: 'row',
      symmetric: true
    },
    status: {overflow: view('uint32', 1)},
    chunks: inputRows.map(rowCount => {
      const chunk = {rowStart, rowCount, weights: createSpatialWeights(rowCount, inputCapacity)};
      rowStart += rowCount;
      return chunk;
    })
  };
  const output = (
    rowCounts = [1, 2, 3],
    capacity = 10
  ): GPUChunkedSpatialWeightsTransposeOutput => {
    let nextRow = 0;
    return {
      chunks: rowCounts.map(rowCount => {
        const chunk = {
          rowStart: nextRow,
          rowCount,
          weights: {
            ...createSpatialWeights(rowCount, capacity),
            status: {
              count: view('uint32', 1),
              requiredCount: view('uint32', 1),
              overflow: view('uint32', 1)
            }
          }
        };
        nextRow += rowCount;
        return chunk;
      })
    };
  };
  return {graph, view, createSpatialWeights, weights, output};
}

it('accepts independent output ranges, preserves descriptions and emits valid bounded kernels', () => {
  const {graph, weights, output, view} = createSetup();
  const destination = {...output(), asymmetricSlots: view('uint32', 1)};
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
  expect(transpose.outputWeights.rowCount).toBe(6);
  expect(transpose.outputWeights.status).toBe(weights.status);
  expect(transpose.outputWeights.spatialContext).toBe(weights.spatialContext);
  expect(transpose.outputWeights.information).toMatchObject({
    transform: 'unknown',
    symmetric: 'unknown'
  });
  for (const [index, chunk] of transpose.outputWeights.chunks.entries()) {
    expect(chunk.weights.offsets).toBe(destination.chunks[index].weights.offsets);
    expect(chunk.weights.status).toBe(destination.chunks[index].weights.status);
    expect(chunk.weights.spatialContext).toBe(weights.spatialContext);
    expect(chunk.weights.information?.rowCount).toBe(chunk.rowCount);
  }
  const nodes = transpose.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(transpose.estimatedCommandNodeCount);
  expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(auditWGSLModule(source), id).toEqual([]);
  }
  graph.device.destroy();
});

it('keeps every binding below the device limit when the total output exceeds it', () => {
  const {graph, weights, output} = createSetup([2, 2, 2]);
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 64});
  const destination = output([2, 2, 2], 10);
  expect(
    destination.chunks.reduce((sum, chunk) => sum + chunk.weights.neighbors.length * 4, 0)
  ).toBeGreaterThan(64);
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
  const nodes = transpose.getCommandNodes(graph);
  for (const node of nodes) {
    for (const resource of node.resources ?? []) {
      if ('buffer' in resource) {
        const handle = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
        expect(handle.byteLength, handle.id).toBeLessThanOrEqual(64);
      }
    }
  }
  graph.device.destroy();
});

it('checks all device binding limits before allocating scratch or creating nodes', () => {
  const {graph, weights, output} = createSetup();
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 32});
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: output([3, 3], 9)});
  expect(() => transpose.getCommandNodes(graph)).toThrow(/input\/output chunk binding.*limit 32/);
  // A failed preflight leaves the first scratch ID free.
  expect(() =>
    createTransientView(graph, `${transpose.id}-output0-column-counts`, 'uint32', 3)
  ).not.toThrow();
  graph.device.destroy();
});

it('validates output ranges, CSR shapes, status ports, distances, aliases and tolerance at construction', () => {
  const {graph, weights, output, view} = createSetup();
  const construct = (destination: GPUChunkedSpatialWeightsTransposeOutput) =>
    new GPUChunkedSpatialWeightsTranspose({id: 'transpose-limits', weights, output: destination});
  expect(() => construct({chunks: []})).toThrow(/transpose-limits output chunk count/);
  const valid = output();
  expect(() => construct({chunks: valid.chunks.slice(0, 2)})).toThrow(/cover columnCount/);
  expect(() =>
    construct({chunks: [{...valid.chunks[0], rowStart: 1}, ...valid.chunks.slice(1)]})
  ).toThrow(/rowStart/);
  expect(() =>
    construct({chunks: [{...valid.chunks[0], rowCount: 2}, ...valid.chunks.slice(1)]})
  ).toThrow(/offsets/);
  expect(() =>
    construct({chunks: [{...valid.chunks[0], rowCount: 0}, ...valid.chunks.slice(1)]})
  ).toThrow(/rowCount/);
  for (const field of ['count', 'requiredCount', 'overflow'] as const) {
    const destination = output();
    // Deliberately test malformed JavaScript callers as well as TypeScript's required fields.
    delete (
      destination.chunks[0].weights.status as Partial<
        (typeof destination.chunks)[0]['weights']['status']
      >
    )[field];
    expect(() => construct(destination)).toThrow(new RegExp(`status.${field} is required`));
    const wrongLength = output();
    wrongLength.chunks[0].weights.status[field] = view('uint32', 2);
    expect(() => construct(wrongLength)).toThrow(/one uint32/);
  }
  const aliased = output();
  aliased.chunks[1].weights.weights = aliased.chunks[0].weights.weights;
  expect(() => construct(aliased)).toThrow(/separate buffers/);
  const inPlace = output();
  inPlace.chunks[0].weights.neighbors = weights.chunks[0].weights.neighbors;
  inPlace.chunks[0].weights.weights = weights.chunks[0].weights.weights;
  inPlace.chunks[0].weights.distances = weights.chunks[0].weights.distances;
  expect(() => construct(inPlace)).toThrow(/share buffers with inputs/);
  const withoutDistances = {
    ...weights,
    chunks: weights.chunks.map(chunk => ({
      ...chunk,
      weights: {...chunk.weights, distances: undefined}
    }))
  };
  expect(
    () => new GPUChunkedSpatialWeightsTranspose({weights: withoutDistances, output: output()})
  ).toThrow(/requires distances/);
  for (const tolerance of [-1, NaN, Infinity, 1e50]) {
    expect(
      () =>
        new GPUChunkedSpatialWeightsTranspose({
          weights,
          output: output(),
          symmetryTolerance: tolerance
        })
    ).toThrow(/symmetryTolerance/);
  }
  expect(
    () =>
      new GPUChunkedSpatialWeightsTranspose({
        weights,
        columnCount: 7,
        output: {...output([7]), asymmetricSlots: view('uint32', 1)}
      })
  ).toThrow(/square/);
  graph.device.destroy();
});

it('refuses output views from another graph', () => {
  const first = createSetup();
  const second = createSetup();
  const transpose = new GPUChunkedSpatialWeightsTranspose({
    weights: first.weights,
    output: second.output()
  });
  expect(() => transpose.getCommandNodes(first.graph)).toThrow(/target graph/);
  first.graph.device.destroy();
  second.graph.device.destroy();
});

it('enforces the node budget at construction and bounds radix-sort and symmetry node counts', () => {
  const {graph, weights, output, view} = createSetup([137, 137, 137], 1025);
  const destination = {...output([127, 129, 155], 2048), asymmetricSlots: view('uint32', 1)};
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
  const nodes = transpose.getCommandNodes(graph);
  expect(nodes.some(node => node.id.includes('radix'))).toBe(true);
  expect(nodes.length).toBeLessThanOrEqual(transpose.estimatedCommandNodeCount);
  expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  // Many one-row output chunks make the repeated sorts exceed the named shared budget.
  expect(
    () =>
      new GPUChunkedSpatialWeightsTranspose({
        weights,
        output: output(Array.from({length: 411}, () => 1))
      })
  ).toThrow(/estimated command node count.*32768/);
  graph.device.destroy();
});

for (const statusLevel of ['matrix', 'chunk'] as const) {
  for (const outputField of ['count', 'requiredCount', 'overflow', 'asymmetricSlots'] as const) {
    it(`rejects output ${outputField} aliasing preserved ${statusLevel} input status`, () => {
      const {graph, weights, output, view} = createSetup();
      const inputStatus = {overflow: view('uint32', 1)};
      if (statusLevel === 'matrix') weights.status = inputStatus;
      else weights.chunks[0].weights.status = inputStatus;
      const destination = output();
      if (outputField === 'asymmetricSlots') destination.asymmetricSlots = inputStatus.overflow;
      else destination.chunks[0].weights.status[outputField] = inputStatus.overflow;
      try {
        expect(() => new GPUChunkedSpatialWeightsTranspose({weights, output: destination})).toThrow(
          /outputs must not share buffers with inputs/
        );
      } finally {
        graph.device.destroy();
      }
    });
  }

  it(`rejects packed output aliasing preserved ${statusLevel} input status`, () => {
    const {graph, weights, view, createSpatialWeights} = createSetup();
    const inputStatus = {overflow: view('uint32', 1)};
    if (statusLevel === 'matrix') weights.status = inputStatus;
    else weights.chunks[0].weights.status = inputStatus;
    try {
      expect(
        () =>
          new GPUChunkedSpatialWeightsTranspose({
            weights,
            output: {...createSpatialWeights(6, 16), asymmetricSlots: inputStatus.overflow}
          })
      ).toThrow(/outputs must not share buffers with inputs/);
    } finally {
      graph.device.destroy();
    }
  });
}

for (const layout of ['chunked', 'packed'] as const) {
  for (const bindingLocation of ['input', 'output'] as const) {
    it(`preflights aligned ${bindingLocation} binding ranges before allocating ${layout} scratch and allows retry`, () => {
      const {graph, weights, output, view, createSpatialWeights} = createSetup([6], 16);
      Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 64});
      const destination = layout === 'chunked' ? output([3, 3], 16) : createSpatialWeights(6, 16);
      const backing = view('uint32', 17);
      const unaligned = graph.createDataView(backing.buffer, {
        format: 'uint32',
        length: 16,
        byteOffset: 4
      });
      const target =
        bindingLocation === 'input'
          ? weights.chunks[0].weights
          : 'chunks' in destination
            ? destination.chunks[1].weights
            : destination;
      target.neighbors = unaligned;
      const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
      // Retain the description before failure, as a downstream consumer may do.
      const description = transpose.outputWeights;
      const describedWeights =
        'chunks' in description ? description.chunks[1].weights : description;
      const expectedWeights = 'chunks' in destination ? destination.chunks[1].weights : destination;
      try {
        expect(() => transpose.getCommandNodes(graph)).toThrow(/68.*64/);
        target.neighbors = view('uint32', 16);
        // The same node/resource IDs must be available after a rejected preflight.
        let nodes: ReturnType<typeof transpose.getCommandNodes<void>> = [];
        expect(() => {
          nodes = transpose.getCommandNodes(graph);
        }).not.toThrow();
        expect(transpose.outputWeights).toBe(description);
        for (const field of ['offsets', 'neighbors', 'weights', 'distances'] as const) {
          expect(describedWeights[field], field).toBe(expectedWeights[field]);
        }
        const scatterWritesCorrectedBuffer = nodes.some(
          node =>
            node.id.endsWith('-scatter') &&
            (node.resources ?? []).some(resource => {
              if (!('buffer' in resource)) return false;
              const buffer = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
              return buffer === describedWeights.neighbors.buffer;
            })
        );
        expect(scatterWritesCorrectedBuffer).toBe(true);
      } finally {
        graph.device.destroy();
      }
    });
  }
}

it('does not claim double standardization for capacity-bounded output chunks', () => {
  const {graph, weights, output, createSpatialWeights} = createSetup();
  weights.information = {...weights.information!, transform: 'double'};
  try {
    const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: output([6], 1)});
    expect(transpose.outputWeights.information?.transform).toBe('unknown');
    expect(transpose.outputWeights.chunks[0].weights.information?.transform).toBe('unknown');
    const packed = new GPUChunkedSpatialWeightsTranspose({
      weights,
      output: createSpatialWeights(6, 16)
    });
    expect(packed.outputWeights.information?.transform).toBe('double');
  } finally {
    graph.device.destroy();
  }
});

it('keeps a saved output description connected to a corrected status port on retry', () => {
  const {graph, weights, output, view} = createSetup([6], 16);
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 64});
  const destination = output([6], 16);
  const backing = view('uint32', 17);
  destination.chunks[0].weights.status.count = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 64
  });
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
  const description = transpose.outputWeights.chunks[0].weights;
  try {
    expect(() => transpose.getCommandNodes(graph)).toThrow(/68.*64/);
    destination.chunks[0].weights.status = {
      ...destination.chunks[0].weights.status,
      count: view('uint32', 1)
    };
    expect(() => transpose.getCommandNodes(graph)).not.toThrow();
    expect(description.status?.count).toBe(destination.chunks[0].weights.status.count);
    expect(description.status).toBe(destination.chunks[0].weights.status);
  } finally {
    graph.device.destroy();
  }
});
