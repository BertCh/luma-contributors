// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState,
  type RasterGatedRound
} from './cost-distance/raster-relaxation';
import {validateRasterBuffersDistinct, validateRasterGrid} from './raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {GPUIterationStatusPort} from '../analysis-core/contracts/status';

/** Shared exact-length and ownership checks for bounded raster producers. @internal */
export function validateRasterExpansionViews(
  id: string,
  grid: GPURasterMetadata,
  inputs: readonly (GraphDataView | undefined)[],
  outputs: readonly (GraphDataView | undefined)[]
): number {
  const count = validateRasterGrid(id, grid);
  for (const view of [...inputs, ...outputs]) {
    if (view) validateSinglePackedView(view, [view.format], id);
  }
  validateRasterBuffersDistinct(id, outputs, inputs);
  return count;
}

/** @internal */
export function validateRasterExpansionGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
}

/** @internal */
export function requireRasterLength(
  view: GraphDataView | undefined,
  length: number,
  id: string = 'raster expansion',
  name: string = 'view'
): void {
  // Every present output must cover the requested dense rows.
  validateRequirement(
    id,
    !view || view.length >= length,
    `${name} must hold at least ${length} entries`
  );
}

/**
 * Status words a bounded iteration publishes, each one `uint32` entry: the named
 * `GPUIterationStatusPort` of `analysis-core/contracts/status`, with `converged` required.
 */
export type RasterIterationStatusOutputs = GPUIterationStatusPort & {
  /** 1 when a round changed nothing before the iteration limit, else 0. */
  converged: GraphDataView<'uint32'>;
  /** 1 when the iteration limit was reached without convergence, else 0. */
  iterationLimitReached?: GraphDataView<'uint32'>;
  /** Number of rounds that ran, including the final round that verified the fixed point. */
  iterationCount?: GraphDataView<'uint32'>;
};

/**
 * Upper bound on the command nodes `getRasterBufferBandNodes` adds for one band that needs
 * canonicalization (texture capture or neighborhood copy), used only for the node budget.
 */
export const RASTER_BAND_CANONICALIZATION_NODE_COUNT = 8;

/** Rounds that share one GPU gate node; rounds after convergence in a group only repeat a fixed point. */
const RASTER_JACOBI_GROUP_SIZE = 4;

/**
 * Number of command nodes of {@link createRasterJacobiLoopNodes} for `roundCount` rounds: the
 * reset node, the slot-table node, the rounds, one gate per group of rounds and the status node.
 * @internal
 */
export function getRasterJacobiLoopNodeCount(roundCount: number): number {
  return 3 + roundCount + Math.ceil(roundCount / RASTER_JACOBI_GROUP_SIZE);
}

/**
 * Unrolls `roundCount` synchronous rounds that stop dispatching on the GPU once a round changes
 * nothing (library rule 13). `createRound` builds the kernel of each round: it must bind
 * `slotBinding` and `status` (`atomic<u32>` read-write, `statusView`), pass `condition` and
 * `extraResources` through, and run `markChangedWGSL` when the round changed any cell.
 * Rounds ping-pong between two buffers; a converged round leaves both buffers equal, so the buffer
 * written by the final unrolled round holds the result whether or not later rounds were skipped.
 * @internal
 */
export function createRasterJacobiLoopNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    invocationCount: number;
    roundCount: number;
    status: RasterIterationStatusOutputs;
    createRound: (
      round: RasterGatedRound<Parameters>,
      statusView: GraphDataView<'uint32'>
    ) => GPUCommandNode<Parameters>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation} = props;
  const state = createRasterIterationState<Parameters>(
    graph,
    `${id}-rounds`,
    operation,
    props.invocationCount,
    id
  );
  const bindings: WGSLKernelBinding[] = [
    {name: 'status', view: state.status, type: 'u32', access: 'read'}
  ];
  for (const name of ['converged', 'iterationLimitReached', 'iterationCount'] as const) {
    const view = props.status[name];
    if (view) bindings.push({name, view, type: 'u32', access: 'read_write'});
  }
  return [
    createRasterIterationResetNode<Parameters>(graph, {id: `${id}-reset`, operation, state}),
    ...createRasterGatedLoopNodes<Parameters>(graph, {
      id: `${id}-round`,
      gateId: `${id}-gate`,
      operation,
      state,
      maxIterations: props.roundCount,
      groupSize: RASTER_JACOBI_GROUP_SIZE,
      createRound: round => props.createRound(round, state.status)
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-status`,
      operation,
      variant: 'iteration-status',
      invocationCount: 1,
      bindings,
      body: `converged[convergedOffset] = status[statusOffset + 3u];
  ${props.status.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = 1u - status[statusOffset + 3u];' : ''}
  ${props.status.iterationCount ? 'iterationCount[iterationCountOffset] = status[statusOffset + 2u];' : ''}`
    })
  ];
}
