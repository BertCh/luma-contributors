// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, getViewBindingRange, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
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
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {GPULocalOutlierFactor} from '../../../src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor';
import {GPUSegregation} from '../../../src/gpu-spatial-analysis/segregation/gpu-segregation';
import {getGPUCommandNodePipelineKeys} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {executeWeightsAuditKernel} from './weights-audit-executor';
import {snapshotWeightsPort} from '../../../src/gpu-spatial-analysis/spatial-weights/weights-construction-snapshot';

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

it('AMG-05 packed transpose forwards loss but omits input cardinalities', () => {
  const {graph, weights, view, createSpatialWeights} = createSetup([2], 2);
  weights.status = {
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  };
  try {
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights,
      columnCount: 1,
      output: createSpatialWeights(1, 2)
    });
    // Input row 0 -> column 0 survives, row 1 -> column 1 would be dropped.
    // Input cardinality 2 therefore cannot describe retained output cardinality 1.
    expect(transpose.outputWeights.status?.count).toBeUndefined();
    expect(transpose.outputWeights.status?.requiredCount).toBeUndefined();
    expect(transpose.outputWeights.status?.overflow).toBe(weights.status.overflow);
    expect(transpose.outputWeights.status?.candidateOverflow).toBe(
      weights.status.candidateOverflow
    );
  } finally {
    graph.device.destroy();
  }
});

it('TSC-03 allocates aggregate status on a graph with frame parameters', () => {
  const graph = new GPUCommandGraph<{frame: number}>(createNullWebGPUDevice());
  const weights = {
    offsets: createTransientView(graph, 'typed-input-offsets', 'uint32', 2),
    neighbors: createTransientView(graph, 'typed-input-neighbors', 'uint32', 1),
    weights: createTransientView(graph, 'typed-input-weights', 'float32', 1)
  };
  const output = {
    offsets: createTransientView(graph, 'typed-output-offsets', 'uint32', 2),
    neighbors: createTransientView(graph, 'typed-output-neighbors', 'uint32', 1),
    weights: createTransientView(graph, 'typed-output-weights', 'float32', 1),
    status: {
      count: createTransientView(graph, 'typed-count', 'uint32', 1),
      requiredCount: createTransientView(graph, 'typed-required', 'uint32', 1),
      overflow: createTransientView(graph, 'typed-overflow', 'uint32', 1)
    }
  };
  try {
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: {rowCount: 1, chunks: [{rowStart: 0, rowCount: 1, weights}]},
      output: {chunks: [{rowStart: 0, rowCount: 1, weights: output}]}
    });
    // getCommandNodes calls the generic allocator; the status getter is its other caller.
    expect(transpose.getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(transpose.outputWeights.status?.overflow?.buffer.graph).toBe(graph);
  } finally {
    graph.device.destroy();
  }
});

it('accepts independent output ranges, preserves descriptions and emits valid bounded kernels', () => {
  const {graph, weights, output, view} = createSetup();
  const destination = {...output(), asymmetricSlots: view('uint32', 1)};
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
  expect(transpose.outputWeights.rowCount).toBe(6);
  // SSR-105: loss is aggregated on the outer port; the input overflow becomes candidateOverflow.
  expect(transpose.outputWeights.status?.overflow).toBeDefined();
  expect(transpose.outputWeights.status?.overflow).not.toBe(weights.status?.overflow);
  expect(transpose.outputWeights.status?.candidateOverflow).toBeDefined();
  expect(transpose.outputWeights.spatialContext).toBe(weights.spatialContext);
  expect(transpose.outputWeights.information).toMatchObject({
    transform: 'unknown',
    symmetric: 'unknown'
  });
  for (const [index, chunk] of transpose.outputWeights.chunks.entries()) {
    expect(chunk.weights.offsets).toBe(destination.chunks[index].weights.offsets);
    expect(chunk.weights.status?.count).toBe(destination.chunks[index].weights.status.count);
    expect(chunk.weights.status?.requiredCount).toBe(
      destination.chunks[index].weights.status.requiredCount
    );
    expect(chunk.weights.status?.overflow).toBeUndefined();
    expect(chunk.weights.spatialContext).toBe(weights.spatialContext);
    expect(chunk.weights.information?.rowCount).toBe(chunk.rowCount);
  }
  const nodes = transpose.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(transpose.estimatedCommandNodeCount);
  expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
  for (const {id, source, constants} of getGPUCommandNodeKernelSources(nodes)) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(auditWGSLModule(source, constants), id).toEqual([]);
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
      const inputStatusView = view('uint32', 1);
      const inputStatus =
        statusLevel === 'matrix' ? {overflow: inputStatusView} : {requiredCount: inputStatusView};
      if (statusLevel === 'matrix') weights.status = inputStatus;
      else weights.chunks[0].weights.status = inputStatus;
      const destination = output();
      if (outputField === 'asymmetricSlots') destination.asymmetricSlots = inputStatusView;
      else destination.chunks[0].weights.status[outputField] = inputStatusView;
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
    const inputStatusView = view('uint32', 1);
    const inputStatus =
      statusLevel === 'matrix' ? {overflow: inputStatusView} : {requiredCount: inputStatusView};
    if (statusLevel === 'matrix') weights.status = inputStatus;
    else weights.chunks[0].weights.status = inputStatus;
    try {
      expect(
        () =>
          new GPUChunkedSpatialWeightsTranspose({
            weights,
            output: {...createSpatialWeights(6, 16), asymmetricSlots: inputStatusView}
          })
      ).toThrow(/outputs must not share buffers with inputs/);
    } finally {
      graph.device.destroy();
    }
  });
}

for (const layout of ['packed', 'chunked'] as const) {
  it(`B2: ${layout} transpose rejects unsupported root and child evidence`, () => {
    const {graph, weights, output, view, createSpatialWeights} = createSetup();
    const destination = layout === 'packed' ? createSpatialWeights(6, 16) : output();
    const construct = (input: GPUChunkedSpatialWeights) =>
      new GPUChunkedSpatialWeightsTranspose({
        id: 'transpose-evidence',
        weights: input,
        output: destination
      });
    try {
      for (const field of [
        'requiredCount',
        'converged',
        'iterationLimitReached',
        'unresolvedCount'
      ] as const) {
        const status = {[field]: view('uint32', 1)};
        expect(() => construct({...weights, status})).toThrow(
          new RegExp(`transpose-evidence.*weights\\.status\\.${field}`)
        );
        const [first, second] = weights.chunks;
        expect(() =>
          construct({
            ...weights,
            status: undefined,
            chunks: [first, {...second, weights: {...second.weights, status}}]
          })
        ).toThrow(
          new RegExp(`transpose-evidence.*weights\\.chunks\\[1\\]\\.weights\\.status\\.${field}`)
        );
      }
    } finally {
      graph.device.destroy();
    }
  });
}

for (const loss of ['overflow', 'candidateOverflow'] as const) {
  it(`aggregates chunk-local ${loss} into inherited candidate loss for either output layout`, () => {
    const {graph, weights, output, view, createSpatialWeights} = createSetup();
    weights.chunks[0].weights.status = {[loss]: view('uint32', 1)};
    try {
      for (const destination of [output(), createSpatialWeights(6, 16)]) {
        const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
        const nodes = transpose
          .getCommandNodes(graph)
          .filter(
            node => node.id.includes('-aggregate-') || node.id.includes('-packed-inherited-loss')
          );
        expect(nodes.length).toBeGreaterThan(0);
        const memory = createCPUGraphMemory();
        for (const word of [1, 0]) {
          memory.write(weights.chunks[0].weights.status![loss]!, Uint32Array.of(word));
          executeCommandNodesOnCPU(nodes, memory);
          // Output capacity cannot recover either kind of input loss; both are inherited loss.
          expect(
            Array.from(memory.readUint32(transpose.outputWeights.status!.candidateOverflow!))
          ).toEqual([word]);
          expect(Array.from(memory.readUint32(transpose.outputWeights.status!.overflow!))).toEqual([
            0
          ]);
        }
      }
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
    expect(description.status?.requiredCount).toBe(
      destination.chunks[0].weights.status.requiredCount
    );
    expect(description.status?.overflow).toBeUndefined();
  } finally {
    graph.device.destroy();
  }
});

it('aggregates destination loss separately from inherited loss and clears candidate canaries', () => {
  const {graph, weights, output, view} = createSetup([8], 8);
  const inputOverflow = weights.status!.overflow!;
  const inputCandidateOverflow = view('uint32', 1);
  weights.status = {
    ...weights.status,
    candidateOverflow: inputCandidateOverflow,
    count: view('uint32', 1),
    requiredCount: view('uint32', 1)
  };
  const destination = output(new Array(8).fill(1), 1);
  for (const chunk of destination.chunks) {
    chunk.weights.status.candidateOverflow = view('uint32', 1);
  }
  try {
    const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
    const status = transpose.outputWeights.status!;
    expect(status.count).toBeUndefined();
    expect(status.requiredCount).toBeUndefined();
    const nodes = transpose.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(transpose.estimatedCommandNodeCount);
    const aggregation = nodes.filter(node => node.id.includes('-aggregate-'));
    const candidateClears = nodes.filter(node => node.id.includes('-candidate-clear-'));
    expect(candidateClears).toHaveLength(8);
    const memory = createCPUGraphMemory();
    // Exercise the second seven-input batch as well as inherited loss and recovery.
    for (const [overflow, candidateOverflow, destinationOverflow] of [
      [0, 0, 0],
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 0]
    ]) {
      memory.write(inputOverflow, Uint32Array.of(overflow));
      memory.write(inputCandidateOverflow, Uint32Array.of(candidateOverflow));
      for (const [index, chunk] of destination.chunks.entries()) {
        memory.write(
          chunk.weights.status.overflow,
          Uint32Array.of(index === 7 ? destinationOverflow : 0)
        );
        memory.write(chunk.weights.status.candidateOverflow!, Uint32Array.of(42));
      }
      memory.write(status.overflow!, Uint32Array.of(99));
      memory.write(status.candidateOverflow!, Uint32Array.of(99));
      // Even before the destination canaries are cleared, they cannot imply inherited loss.
      executeCommandNodesOnCPU(aggregation, memory);
      expect(Array.from(memory.readUint32(status.overflow!))).toEqual([destinationOverflow]);
      expect(Array.from(memory.readUint32(status.candidateOverflow!))).toEqual([
        Number(Boolean(overflow || candidateOverflow))
      ]);
      executeCommandNodesOnCPU(candidateClears, memory);
      for (const chunk of destination.chunks) {
        expect(Array.from(memory.readUint32(chunk.weights.status.candidateOverflow!))).toEqual([0]);
      }
    }
    for (const chunk of transpose.outputWeights.chunks) {
      expect(chunk.weights.status?.overflow).toBeUndefined();
      expect(chunk.weights.status?.candidateOverflow).toBeUndefined();
    }
  } finally {
    graph.device.destroy();
  }
});

it('revalidates corrected ports without reallocating the published aggregate status', () => {
  const {graph, weights, output, view} = createSetup();
  const destination = output();
  destination.chunks[0].weights.status.candidateOverflow = view('uint32', 1);
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
  const status = transpose.outputWeights.status!;
  const distances = weights.chunks[0].weights.distances;
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    weights.chunks[0].weights.distances = undefined;
    expect(() => transpose.getCommandNodes(graph)).toThrow(/requires distances/);
    expect(allocations).not.toHaveBeenCalled();
    weights.chunks[0].weights.distances = distances;
    destination.chunks[0].weights.status.candidateOverflow =
      destination.chunks[0].weights.status.count;
    expect(() => transpose.getCommandNodes(graph)).toThrow(/separate buffers|overlap/);
    expect(allocations).not.toHaveBeenCalled();
    destination.chunks[0].weights.status.candidateOverflow = view('uint32', 1);
    expect(() => transpose.getCommandNodes(graph)).not.toThrow();
    expect(transpose.outputWeights.status?.overflow).toBe(status.overflow);
    expect(transpose.outputWeights.status?.candidateOverflow).toBe(status.candidateOverflow);
  } finally {
    allocations.mockRestore();
    graph.device.destroy();
  }
});

// Merge review regressions live here because this is the stream's owned Node spec.
for (const correctedPort of ['output count', 'input loss'] as const) {
  it(`B1: refuses ${correctedPort} aliasing a published aggregate before scratch and allows correction`, () => {
    const {graph, weights, output} = createSetup();
    const destination = output();
    const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
    const status = transpose.outputWeights.status!;
    const count = destination.chunks[0].weights.status.count;
    const inputLoss = weights.status!.overflow!;
    if (correctedPort === 'output count')
      destination.chunks[0].weights.status.count = status.overflow!;
    else weights.status!.overflow = status.candidateOverflow!;
    const allocations = vi.spyOn(graph, 'createTransientBuffer');
    try {
      expect(() => transpose.getCommandNodes(graph)).toThrow(
        /aggregate.*share buffers|share buffers.*aggregate/
      );
      expect(allocations).not.toHaveBeenCalled();
      destination.chunks[0].weights.status.count = count;
      weights.status!.overflow = inputLoss;
      const nodes = transpose.getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(count, Uint32Array.of(2));
      memory.write(inputLoss, Uint32Array.of(0));
      destination.chunks.forEach(chunk =>
        memory.write(chunk.weights.status.overflow, Uint32Array.of(0))
      );
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.includes('-aggregate-')),
        memory
      );
      expect(Array.from(memory.readUint32(count))).toEqual([2]);
      expect(Array.from(memory.readUint32(status.overflow!))).toEqual([0]);
      expect(Array.from(memory.readUint32(status.candidateOverflow!))).toEqual([0]);
      expect(transpose.outputWeights.status?.overflow).toBe(status.overflow);
    } finally {
      allocations.mockRestore();
      graph.device.destroy();
    }
  });
}

it('B2: chooses the corrected graph for unpublished aggregate handles', () => {
  const setup = createSetup();
  const foreign = createSetup();
  const destination = setup.output();
  const original = destination.chunks[0].weights.offsets;
  destination.chunks[0].weights.offsets = foreign.view('uint32', original.length);
  const transpose = new GPUChunkedSpatialWeightsTranspose({
    weights: setup.weights,
    output: destination
  });
  const allocations = vi.spyOn(setup.graph, 'createTransientBuffer');
  try {
    expect(() => transpose.getCommandNodes(setup.graph)).toThrow(/target graph/);
    expect(allocations).not.toHaveBeenCalled();
    destination.chunks[0].weights.offsets = original;
    expect(() => transpose.getCommandNodes(setup.graph)).not.toThrow();
    expect(transpose.outputWeights.status?.overflow?.buffer.graph).toBe(setup.graph);
    expect(transpose.outputWeights.status?.candidateOverflow?.buffer.graph).toBe(setup.graph);
  } finally {
    allocations.mockRestore();
    setup.graph.device.destroy();
    foreign.graph.device.destroy();
  }
});

it('B2: rejects an already-published aggregate on another graph before scratch', () => {
  const setup = createSetup();
  const replacement = createSetup();
  const destination = setup.output();
  const transpose = new GPUChunkedSpatialWeightsTranspose({
    weights: setup.weights,
    output: destination
  });
  const status = transpose.outputWeights.status!;
  const originalChunks = setup.weights.chunks;
  const originalStatus = setup.weights.status;
  const outputChunks = destination.chunks;
  setup.weights.chunks = replacement.weights.chunks;
  setup.weights.status = replacement.weights.status;
  destination.chunks = replacement.output().chunks;
  const allocations = vi.spyOn(replacement.graph, 'createTransientBuffer');
  try {
    expect(() => transpose.getCommandNodes(replacement.graph)).toThrow(/target graph/);
    expect(allocations).not.toHaveBeenCalled();
    setup.weights.chunks = originalChunks;
    setup.weights.status = originalStatus;
    destination.chunks = outputChunks;
    expect(() => transpose.getCommandNodes(setup.graph)).not.toThrow();
    expect(transpose.outputWeights.status?.overflow).toBe(status.overflow);
  } finally {
    allocations.mockRestore();
    setup.graph.device.destroy();
    replacement.graph.device.destroy();
  }
});

it('B3: publishes inherited chunk ranges and live inherited cardinalities', () => {
  const {graph, weights, output, view} = createSetup();
  const destination = output();
  const original = destination.chunks[0];
  const inheritedStatus = Object.create(original.weights.status) as typeof original.weights.status;
  destination.chunks = [
    Object.create({
      rowStart: original.rowStart,
      rowCount: original.rowCount,
      weights: {...original.weights, status: inheritedStatus}
    }) as typeof original,
    ...destination.chunks.slice(1)
  ];
  try {
    const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output: destination});
    const described = transpose.outputWeights.chunks[0];
    expect(described.rowStart).toBe(0);
    expect(described.rowCount).toBe(1);
    expect(described.weights.status?.count).toBe(original.weights.status.count);
    expect(described.weights.status?.requiredCount).toBe(original.weights.status.requiredCount);
    const correctedCount = view('uint32', 1);
    destination.chunks[0].weights.status = Object.create({
      ...original.weights.status,
      count: correctedCount
    });
    expect(described.weights.status?.count).toBe(correctedCount);
    expect(described.weights.status?.overflow).toBeUndefined();
    expect(described.weights.status?.candidateOverflow).toBeUndefined();
    expect(() => transpose.getCommandNodes(graph)).not.toThrow();
  } finally {
    graph.device.destroy();
  }
});

for (const layout of ['packed', 'chunked'] as const) {
  it(`B3: rejects inherited unsupported ${layout} destination status at construction`, () => {
    const {graph, weights, output, view, createSpatialWeights} = createSetup();
    const invalidCount = view('uint32', 1);
    const destination = output();
    destination.chunks[0].weights.status = Object.create({
      ...destination.chunks[0].weights.status,
      invalidCount
    });
    try {
      expect(
        () =>
          new GPUChunkedSpatialWeightsTranspose({
            weights,
            output:
              layout === 'chunked'
                ? destination
                : {
                    ...createSpatialWeights(6, 16),
                    status: Object.create({invalidCount})
                  }
          })
      ).toThrow(/output.status|status.invalidCount/);
    } finally {
      graph.device.destroy();
    }
  });
}

function createReviewMarkov() {
  const setup = createSetup([2], 2);
  const {view, graph} = setup;
  const convergedStorage = view('uint32', 2);
  // Status ports are exactly one word. Keep a larger backing view to test the untouched tail.
  const converged = graph.createDataView(convergedStorage.buffer, {format: 'uint32', length: 1});
  return {
    ...setup,
    convergedStorage,
    props: {
      values: view('float32', 4),
      classes: view('uint32', 4),
      classCount: 2,
      weights: setup.weights.chunks[0].weights,
      periodCount: 2,
      maximumLagClassCount: 2,
      lagParameters: view('float32', 16),
      output: {
        counts: view('uint32', 10),
        probabilities: view('float32', 10),
        rowTotals: view('uint32', 6),
        lagBreaks: view('float32', 5),
        lagClassCount: view('uint32', 2),
        homogeneity: view('float32', 7),
        steadyStates: view('float32', 8),
        status: {converged}
      }
    }
  };
}

it('B4: spatial Markov loss guards preserve every oversized output tail across recovery', () => {
  const {graph, props, view, convergedStorage} = createReviewMarkov();
  const overflow = view('uint32', 1);
  const weights = {...props.weights, status: {overflow}};
  try {
    const nodes = new GPUSpatialMarkov({...props, weights}).getCommandNodes(graph);
    const guards = nodes.filter(
      node =>
        node.id.startsWith('spatial-markov-') &&
        !node.id.startsWith('spatial-markov-lag-') &&
        node.id.includes('-incomplete-')
    );
    const outputs = [
      [props.output.counts, 8],
      [props.output.probabilities, 8],
      [props.output.rowTotals, 4],
      [props.output.lagBreaks, 3],
      [props.output.lagClassCount, 1],
      [props.output.homogeneity, 5],
      [props.output.steadyStates, 6],
      [convergedStorage, 1]
    ] as const;
    const memory = createCPUGraphMemory();
    for (const loss of [1, 0]) {
      memory.write(overflow, Uint32Array.of(loss));
      // Simulate ordinary writers restoring only their logical prefix, retaining tail canaries.
      for (const [output, length] of outputs) {
        if (loss)
          memory.write(
            output,
            output.format === 'float32'
              ? new Float32Array(output.length).fill(42)
              : new Uint32Array(output.length).fill(42)
          );
        memory.write(
          output,
          output.format === 'float32'
            ? new Float32Array(length).fill(7)
            : new Uint32Array(length).fill(7)
        );
      }
      executeCommandNodesOnCPU(guards, memory);
      for (const [output, length] of outputs) {
        const actual =
          output.format === 'float32' ? memory.readFloat32(output) : memory.readUint32(output);
        expect(Array.from(actual.slice(length)), output.buffer.id).toEqual(
          new Array(output.length - length).fill(42)
        );
        expect(Array.from(actual.slice(0, length)), output.buffer.id).toEqual(
          new Array(length).fill(loss ? (output.format === 'float32' ? Number.NaN : 0) : 7)
        );
      }
    }
  } finally {
    graph.device.destroy();
  }
});

it('B5: spatial Markov retains prototype-backed props, output routes and convergence status', () => {
  const {graph, props} = createReviewMarkov();
  recordWGSLKernelNodes(graph);
  const output = Object.create({
    ...props.output,
    status: Object.create(props.output.status)
  }) as typeof props.output;
  const inherited = Object.create({...props, output}) as typeof props;
  try {
    const operation = new GPUSpatialMarkov(inherited);
    expect(operation.props.output.status?.converged).toBe(props.output.status.converged);
    expect(operation.props.output.probabilities).toBe(props.output.probabilities);
    expect(
      operation
        .getCommandNodes(graph)
        .some(node =>
          getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
            binding =>
              binding.view === props.output.status.converged && binding.access === 'read_write'
          )
        )
    ).toBe(true);
  } finally {
    graph.device.destroy();
  }
});

it('B5: LOF retains prototype-backed props and optional output routes', () => {
  const {graph, view, createSpatialWeights} = createSetup([2], 2);
  const output = {
    kDistance: view('float32', 2),
    localReachabilityDensity: view('float32', 2),
    lof: view('float32', 2),
    outlier: view('uint32', 2),
    outlierCount: view('uint32', 1)
  };
  try {
    const operation = new GPULocalOutlierFactor(
      Object.create({
        neighbors: createSpatialWeights(2, 2),
        parameters: view('float32', 4),
        output: Object.create(output)
      })
    );
    expect(operation.props.output.outlier).toBe(output.outlier);
    expect(operation.props.output.outlierCount).toBe(output.outlierCount);
    expect(operation.getCommandNodes(graph).some(node => node.id.endsWith('-outlier'))).toBe(true);
  } finally {
    graph.device.destroy();
  }
});

it('B6: segregation retains and validates supported inherited scale status', () => {
  const {graph, view, createSpatialWeights} = createSetup([2], 2);
  const status = {
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1),
    invalidCount: view('uint32', 1)
  };
  const weights = createSpatialWeights(2, 2);
  const props = {
    groupCount: 2,
    groupCounts: view('float32', 4),
    output: {indices: view('float32', 13)}
  };
  try {
    const operation = new GPUSegregation({
      ...props,
      scales: [{...weights, status: Object.create(status)}]
    });
    expect(operation.props.scales?.[0]?.status).toMatchObject(status);
    expect(
      () =>
        new GPUSegregation({
          ...props,
          scales: [
            {...weights, status: Object.create({...status, invalidCount: view('uint32', 2)})}
          ]
        })
    ).toThrow(/scales\[0\]\.status invalidCount must contain one uint32 row/);
    for (const field of ['complete', 'converged', 'iterationLimitReached', 'requiredCount']) {
      expect(
        () =>
          new GPUSegregation({
            ...props,
            scales: [{...weights, status: Object.create({...status, [field]: view('uint32', 1)})}]
          })
      ).toThrow(new RegExp(`segregation scales\\[0\\]\\.status\\.${field} cannot be qualified`));
    }
    for (const inherit of [false, true]) {
      const unresolvedCount = view('uint32', 1);
      const unresolvedStatus = {...status, unresolvedCount};
      const scale = {
        ...weights,
        status: inherit ? Object.create(unresolvedStatus) : unresolvedStatus
      };
      expect(snapshotWeightsPort(scale).status.unresolvedCount).toBe(unresolvedCount);
      expect(() => new GPUSegregation({...props, scales: [scale]})).toThrow(
        /segregation scales\[0\]\.status\.unresolvedCount cannot be qualified/
      );
      const malformed = {...status, unresolvedCount: view('uint32', 2)};
      expect(
        () =>
          new GPUSegregation({
            ...props,
            scales: [{...weights, status: inherit ? Object.create(malformed) : malformed}]
          })
      ).toThrow(/segregation scales\[0\]\.status unresolvedCount must contain one uint32 row/);
    }
  } finally {
    graph.device.destroy();
  }
});

for (const spatialForm of ['environment', 'smoothed-population'] as const) {
  it(`S1: loss-bearing segregation scales share pipelines and address their own blocks (${spatialForm})`, () => {
    const build = (scaleCount: number) => {
      const {graph, view, createSpatialWeights} = createSetup([2], 2);
      const weights = {...createSpatialWeights(2, 2), status: {overflow: view('uint32', 1)}};
      const indices = view('float32', 13 * scaleCount);
      const operation = new GPUSegregation({
        groupCount: 2,
        groupCounts: view('float32', 4),
        spatialForm,
        scales: Array.from({length: scaleCount}, () => weights),
        output: {indices}
      });
      return {graph, indices, weights, nodes: operation.getCommandNodes(graph)};
    };
    const two = build(2),
      three = build(3);
    try {
      expect(getGPUCommandNodePipelineKeys(three.nodes).size).toBe(
        getGPUCommandNodePipelineKeys(two.nodes).size
      );
      const memory = createCPUGraphMemory();
      memory.write(three.weights.status.overflow, Uint32Array.of(1));
      memory.write(three.indices, new Float32Array(39).fill(42));
      // Execute only environment cursor publication (with zero counts), then the last guard.
      executeCommandNodesOnCPU(
        three.nodes.filter(
          node =>
            node.id.endsWith('-scale-cursor-clear') ||
            /^segregation-scale-\d+-environment$/.test(node.id)
        ),
        memory
      );
      executeCommandNodesOnCPU(
        three.nodes.filter(node => node.id === 'segregation-scale-2-incomplete'),
        memory
      );
      const actual = Array.from(memory.readFloat32(three.indices));
      expect(actual.slice(0, 26)).toEqual(new Array(26).fill(42));
      actual.slice(26).forEach((value, index) => {
        if (spatialForm === 'environment' && index === 2) expect(value).toBe(42);
        else expect(value).toBeNaN();
      });
    } finally {
      two.graph.device.destroy();
      three.graph.device.destroy();
    }
  });
}

for (const pairedOutputs of ['kDistance-density', 'density-lof'] as const) {
  it(`S2: LOF supports packed ${pairedOutputs} without conflicting physical-buffer usages`, () => {
    const {graph, view, createSpatialWeights} = createSetup([2], 2);
    const packed = view('float32', 128);
    const second = graph.createDataView(packed.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 256
    });
    const first = graph.createDataView(packed.buffer, {format: 'float32', length: 2});
    const output = {
      kDistance: pairedOutputs === 'kDistance-density' ? first : view('float32', 2),
      localReachabilityDensity: pairedOutputs === 'kDistance-density' ? second : first,
      lof: pairedOutputs === 'density-lof' ? second : view('float32', 2)
    };
    try {
      recordWGSLKernelNodes(graph);
      const operation = new GPULocalOutlierFactor({
        neighbors: createSpatialWeights(2, 2),
        parameters: view('float32', 4),
        output
      });
      const nodes = operation.getCommandNodes(graph);
      for (const node of nodes) {
        const usages = new Map<object, Set<string>>();
        for (const resource of node.resources ?? []) {
          if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) continue;
          const buffer = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
          const current = usages.get(buffer) ?? new Set<string>();
          current.add(resource.usage);
          usages.set(buffer, current);
        }
        for (const current of usages.values())
          expect(current.has('storage-read') && current.has('storage-read-write'), node.id).toBe(
            false
          );
      }
      const memory = createCPUGraphMemory();
      memory.write(operation.props.neighbors.offsets, Uint32Array.of(0, 1, 2));
      memory.write(operation.props.neighbors.neighbors, Uint32Array.of(1, 0));
      memory.write(operation.props.neighbors.distances!, Float32Array.of(1, 1));
      memory.write(operation.props.parameters, Float32Array.of(1, 0.5, 0, 0));
      // WgslExec discards a typed-array binding's nonzero base offset. Copy the actual
      // aligned binding ranges into independent arrays and publish only writable ranges;
      // the production WGSL offsets and arithmetic remain unchanged.
      const kernels = new Map(
        getGPUCommandNodeKernelSources(nodes).map(kernel => [kernel.id, kernel])
      );
      for (const node of nodes) {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        const buffers: Record<string, Float32Array | Uint32Array> = {};
        for (const binding of record.bindings) {
          const range = getViewBindingRange(binding.view);
          const backing = memory.getBuffer(binding.view.buffer).backing;
          const bytes = backing.slice(range.offset, range.offset + range.size);
          buffers[binding.name] =
            binding.view.format === 'float32' ? new Float32Array(bytes) : new Uint32Array(bytes);
        }
        executeWeightsAuditKernel(kernels.get(node.id)!, buffers);
        for (const binding of record.bindings) {
          if (binding.access !== 'read_write') continue;
          const range = getViewBindingRange(binding.view);
          const backing = memory.getBuffer(binding.view.buffer).backing;
          new Uint8Array(backing, range.offset, range.size).set(
            new Uint8Array(buffers[binding.name].buffer)
          );
        }
      }
      expect(Array.from(memory.readFloat32(output.kDistance))).toEqual([1, 1]);
      memory
        .readFloat32(output.localReachabilityDensity)
        .forEach(value => expect(value).toBeCloseTo(2 / 3, 6));
      expect(Array.from(memory.readFloat32(output.lof))).toEqual([1, 1]);
    } finally {
      graph.device.destroy();
    }
  });
}
