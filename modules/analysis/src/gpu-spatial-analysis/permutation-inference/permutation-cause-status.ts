// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWeightsLossNodes,
  getWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import {
  getGPUSpatialWeightsChunks,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  getGPUSpatialWeightsNeighborLoopWGSL,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {createSpatialStatisticsRowNode} from '../spatial-weights/spatial-statistics-row-dispatch';
import {getPermutationParameterWGSL} from './permutation-inference-kernels';

/** Independent causes; every supplied scalar is overwritten on each submission. */
export type GPUPermutationCauseStatus = {
  /** Inherited final weights-capacity loss. */
  weightsOverflow?: GraphDataView<'uint32'>;
  /** Inherited intermediate weights-capacity loss. */
  candidateOverflow?: GraphDataView<'uint32'>;
  /** Invalid runtime draw count, or a local significance cutoff outside float32 (0,1). */
  invalidInput?: GraphDataView<'uint32'>;
};

/** @internal */
export function validatePermutationCauseStatus(
  id: string,
  status: GPUPermutationCauseStatus & {neighborOverflow?: GraphDataView<'uint32'>} = {}
) {
  for (const name of [
    'weightsOverflow',
    'candidateOverflow',
    'invalidInput',
    'neighborOverflow'
  ] as const) {
    const view = status[name];
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      validateCapacity(id, `output.status.${name} length`, view.length, 1, 1);
    }
  }
}

/** Separate kernels avoid changing the legacy aggregate flag or exceeding its binding budget. @internal */
export function getPermutationCauseStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
    parameters: GraphDataView<'uint32'>;
    maximumPermutations: number;
    status?: GPUPermutationCauseStatus & {neighborOverflow?: GraphDataView<'uint32'>};
    rowPositions?: GraphDataView<'uint32'>;
    maximumNeighbors?: number;
    usesSignificanceLevel?: boolean;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, status} = props;
  const sources = getWeightsLossSources([props.weights], id, true);
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (status?.weightsOverflow)
    nodes.push(
      ...createWeightsLossNodes(
        graph,
        `${id}-cause-weights`,
        sources.overflow,
        status.weightsOverflow
      )
    );
  if (status?.candidateOverflow)
    nodes.push(
      ...createWeightsLossNodes(
        graph,
        `${id}-cause-candidates`,
        sources.candidateOverflow,
        status.candidateOverflow
      )
    );
  if (status?.invalidInput)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-cause-settings`,
        operation,
        invocationCount: 1,
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'invalidInput', view: status.invalidInput, type: 'u32', access: 'read_write'}
        ],
        declarations: getPermutationParameterWGSL(props.maximumPermutations),
        body: `invalidInput[invalidInputOffset] = u32(!isPermutationCountValid()${props.usesSignificanceLevel ? ' || !isPermutationSignificanceValid()' : ''});`
      })
    );
  const neighborOverflow = status?.neighborOverflow;
  if (neighborOverflow && props.rowPositions) {
    nodes.push(
      createFillNode(graph, {
        id: `${id}-cause-neighbors-clear`,
        operation,
        view: neighborOverflow,
        type: 'u32',
        value: '0u',
        componentCount: 1
      })
    );
    const rows = props.rowPositions.length - 1;
    for (const [chunkIndex, chunk] of getGPUSpatialWeightsChunks(props.weights).entries())
      nodes.push(
        createSpatialStatisticsRowNode(graph, {
          id: `${id}-cause-neighbors-${chunkIndex}`,
          operation,
          rowBase: chunk.rowStart,
          invocationCount: chunk.rowCount,
          bindings: [
            {name: 'offsets', view: chunk.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: chunk.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: chunk.weights.weights, type: 'f32', access: 'read'},
            {name: 'positions', view: props.rowPositions, type: 'u32', access: 'read'},
            {
              name: 'neighborOverflow',
              view: neighborOverflow,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `if (positions[positionsOffset + index] != 0xffffffffu) {
  var count = 0u;
  ${getGPUSpatialWeightsNeighborLoopWGSL('positions[positionsOffset + neighbor] != 0xffffffffu', 'count++;', 'index', 'WEIGHTS_ROW_START')}
  if (count > ${props.maximumNeighbors ?? 32}u) { atomicMax(&neighborOverflow[neighborOverflowOffset], 1u); }
}`
        })
      );
  }
  return nodes;
}
