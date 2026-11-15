// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUChunkedSpatialWeights} from './gpu-chunked-spatial-weights';

/** Checks the input, independently of output capacity; never packs all rows or slots. @internal */
export function getChunkedSpatialWeightsSymmetryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  weights: GPUChunkedSpatialWeights,
  asymmetricSlots: GraphDataView<'uint32'>,
  tolerance: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const chunkTotals = createTransientView(
    graph,
    `${id}-symmetry-totals`,
    'uint32',
    weights.chunks.length
  );
  weights.chunks.forEach((sourceChunk, sourceIndex) => {
    const chunkId = `${id}-symmetry-${sourceIndex}`;
    const mismatches = createTransientView(
      graph,
      `${chunkId}-rows`,
      'uint32',
      sourceChunk.rowCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${chunkId}-clear`,
        operation,
        variant: 'symmetry-clear',
        bindings: [{name: 'mismatches', view: mismatches, type: 'u32', access: 'read_write'}],
        invocationCount: sourceChunk.rowCount,
        body: 'mismatches[mismatchesOffset + index] = 0u;'
      })
    );
    weights.chunks.forEach((targetChunk, targetIndex) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${chunkId}-target-${targetIndex}`,
          operation,
          variant: 'chunked-symmetry',
          bindings: [
            {name: 'sourceOffsets', view: sourceChunk.weights.offsets, type: 'u32', access: 'read'},
            {
              name: 'sourceNeighbors',
              view: sourceChunk.weights.neighbors,
              type: 'u32',
              access: 'read'
            },
            {name: 'sourceWeights', view: sourceChunk.weights.weights, type: 'f32', access: 'read'},
            {name: 'targetOffsets', view: targetChunk.weights.offsets, type: 'u32', access: 'read'},
            {
              name: 'targetNeighbors',
              view: targetChunk.weights.neighbors,
              type: 'u32',
              access: 'read'
            },
            {name: 'targetWeights', view: targetChunk.weights.weights, type: 'f32', access: 'read'},
            {name: 'mismatches', view: mismatches, type: 'u32', access: 'read_write'},
            {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'}
          ],
          invocationCount: sourceChunk.rowCount,
          declarations: `${getBinarySearchWGSL('sourceNeighbors')}
${getBinarySearchWGSL('targetNeighbors')}
const SOURCE_START: u32 = ${sourceChunk.rowStart}u;
const TARGET_START: u32 = ${targetChunk.rowStart}u;
const TARGET_END: u32 = ${targetChunk.rowStart + targetChunk.rowCount}u;`,
          // Each present directed slot contributes one mismatch when its reverse differs, and
          // two when its reverse is absent (both positions of the union differ).
          body: `let row = SOURCE_START + index;
  let begin = sourceOffsets[sourceOffsetsOffset + index];
  let end = sourceOffsets[sourceOffsetsOffset + index + 1u];
  let first = lowerBoundSourceNeighbors(begin, end, TARGET_START);
  let last = lowerBoundSourceNeighbors(first, end, TARGET_END);
  var count = 0u;
  for (var slot = first; slot < last; slot++) {
    let neighbor = sourceNeighbors[sourceNeighborsOffset + slot] - TARGET_START;
    let reverseBegin = targetOffsets[targetOffsetsOffset + neighbor];
    let reverseEnd = targetOffsets[targetOffsetsOffset + neighbor + 1u];
    let reverse = lowerBoundTargetNeighbors(reverseBegin, reverseEnd, row);
    if (reverse < reverseEnd && targetNeighbors[targetNeighborsOffset + reverse] == row) {
      count += u32(abs(sourceWeights[sourceWeightsOffset + slot] - targetWeights[targetWeightsOffset + reverse]) > tolerance[toleranceOffset]);
    } else {
      count += 2u;
    }
  }
  mismatches[mismatchesOffset + index] += count;`
        })
      );
    });
    const total = graph.createDataView(chunkTotals.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: sourceIndex * Uint32Array.BYTES_PER_ELEMENT
    });
    nodes.push(
      ...new GPUReduction({
        id: `${chunkId}-total`,
        input: mismatches,
        output: total,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  });
  nodes.push(
    ...new GPUReduction({
      id: `${id}-symmetry-total`,
      input: chunkTotals,
      output: asymmetricSlots,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  return nodes;
}
