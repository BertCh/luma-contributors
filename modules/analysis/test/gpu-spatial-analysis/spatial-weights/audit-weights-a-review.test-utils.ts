// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULatticeWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-lattice-weights';
import {createCSRStages} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-algebra-wgsl';
import {
  createWGSLKernelNode,
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';

/** Small host/interpreter rig. This is not GPU execution. */
export function createReviewSetup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const getBuffer = memory.getBuffer;
  memory.getBuffer = logical => {
    const buffer = getBuffer(logical);
    if (buffer.backing.byteLength > logical.byteLength)
      buffer.backing = buffer.backing.slice(0, logical.byteLength);
    return buffer;
  };
  let serial = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    data: Uint32Array | Float32Array,
    format: Format
  ) => {
    const view = createTransientView(
      graph,
      `review-${serial++}`,
      format,
      data.length / (format === 'float32x2' ? 2 : 1)
    );
    memory.write(view, data);
    return view;
  };
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    input(
      format === 'float32' ? new Float32Array(length).fill(-999) : new Uint32Array(length).fill(99),
      format
    );
  const weights = (rows = 2, capacity = 2, distances = false) => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined
  });
  const output = (rows = 2, capacity = 2, distances = false) => ({
    ...weights(rows, capacity, distances),
    status: {
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    }
  });
  return {device, graph, memory, input, view, weights, output};
}

/** Exercise production prefix/offset stages with synthetic counts on both sides of a u32 carry. */
export function createReviewCarryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  kind: 'lattice' | 'algebra',
  output: {
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
    status: {overflow: GraphDataView<'uint32'>; requiredCount: GraphDataView<'uint32'>};
  }
) {
  const id = `review-carry-${kind}`;
  recordWGSLKernelNodes(graph);
  const nodes =
    kind === 'lattice'
      ? new GPULatticeWeights({
          id,
          width: 3,
          height: 1,
          criterion: 'queen',
          output
        }).getCommandNodes(graph)
      : createCSRStages(graph, {
          id,
          operation: 'ReviewAlgebraCarry',
          rows: 3,
          targetOffsets: output.offsets,
          capacity: 4,
          overflow: output.status.overflow,
          totalNeighbors: output.status.requiredCount,
          initializesOverflow: true,
          bindings: [],
          visit: () => '',
          attributes: []
        });
  const counts = getWGSLKernelNodeRecord(graph, `${id}-counts`)!.bindings.find(
    binding => binding.name === 'counts'
  )!.view;
  // Replace only candidate enumeration; all scan, saturation and clamped-offset stages stay real.
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-synthetic-counts`,
      operation: 'ReviewCarry',
      invocationCount: 3,
      bindings: [{name: 'counts', view: counts, type: 'u32', access: 'read_write'}],
      body: 'counts[countsOffset + index] = select(select(7u, 3u, index == 1u), 4294967294u, index == 0u);'
    }),
    ...nodes.filter(node => node.id !== `${id}-counts` && !node.id.startsWith(`${id}-emit`))
  ];
}
