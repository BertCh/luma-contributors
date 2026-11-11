// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGPUStatusPort} from '../analysis-core/contracts/status';
import {
  getGraphOutputViews,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionGraph,
  type RasterIterationStatusOutputs
} from './raster-expansion-utils';
import {
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  getRasterIterationCondition
} from './cost-distance/raster-relaxation';
import {GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION} from './distance-field/distance-field-parameters';
import {getRasterMedialAxisNodes} from './raster-medial-axis';

/** Default number of thinning iterations, each made of two Zhang-Suen sub-iterations. */
const DEFAULT_MAXIMUM_ITERATIONS = 64;

export type GPURasterSkeletonProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GraphDataView<'uint32'>;
  /** Thinning iterations before the run stops unconverged. Default 64; at most 10,920 (the command node budget). */
  maximumIterations?: number;
  output: {skeleton: GraphDataView<'uint32'>; status: RasterIterationStatusOutputs};
};
/**
 * Zhang–Suen two-subiteration thinning. Foreground is nonzero; the exterior is background.
 * The conditions are those of Zhang and Suen (1984) unchanged, so the result has the known
 * properties of that algorithm: an isolated 2 x 2 block is deleted completely (each of its pixels
 * has three neighbours and one 0-to-1 transition, and each sub-iteration removes two of them),
 * two-pixel-thick strokes can vanish, and the ends of a thick bar are eroded. Use
 * `GPURasterMedialAxis` when every component must keep a centre.
 * Iterations after convergence dispatch no work, and the current skeleton is always published;
 * check `status.converged` to know whether thinning finished.
 */
export class GPURasterSkeleton implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterSkeletonProps;
  /** Command nodes `getCommandNodes` returns; checked against the node budget. */
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterSkeletonProps) {
    this.id = props.id ?? 'raster-skeleton';
    this.props = props;
    const {id} = this;
    const count = validateRasterExpansionViews(
      id,
      props.grid,
      [props.input],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.input, count, id, 'input');
    requireRasterLength(props.output.skeleton, count, id, 'output.skeleton');
    requireRasterLength(props.output.status.converged, 1, id, 'output.status.converged');
    requireRasterLength(
      props.output.status.iterationLimitReached,
      1,
      id,
      'output.status.iterationLimitReached'
    );
    requireRasterLength(props.output.status.iterationCount, 1, id, 'output.status.iterationCount');
    validateGPUStatusPort(id, props.output.status);
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateRequirement(
      id,
      Number.isInteger(maximumIterations) && maximumIterations > 0,
      'maximumIterations must be a positive integer'
    );
    // Initialize, reset, status and publish, plus two sub-iterations and one gate per iteration.
    this.estimatedCommandNodeCount = 4 + 3 * (maximumIterations + 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [props.input, ...getGraphOutputViews(output)]);
    const first = createTransientView(graph, `${id}-first`, 'uint32', count);
    const second = createTransientView(graph, `${id}-second`, 'uint32', count);
    const operation = 'GPURasterSkeleton';
    const roundCount = (props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS) + 1;
    const state = createRasterIterationState<Parameters>(
      graph,
      `${id}-rounds`,
      operation,
      count,
      id
    );
    const declarations = `const WIDTH: u32 = ${grid.width}u;
const HEIGHT: u32 = ${grid.height}u;

// Value of the neighbor on side 0..7 (clockwise from north); outside the raster is background.
fn getNeighborValue(index: u32, side: u32) -> u32 {
  let columns = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let rows = array<i32, 8>(-1, -1, 0, 1, 1, 1, 0, -1);
  let column = i32(index % WIDTH) + columns[side];
  let row = i32(index / WIDTH) + rows[side];
  if (column < 0 || row < 0 || column >= i32(WIDTH) || row >= i32(HEIGHT)) { return 0u; }
  return previous[previousOffset + u32(row) * WIDTH + u32(column)];
}

// Zhang-Suen deletion test for sub-iteration phase 0 or 1.
fn shouldRemove(index: u32, phase: u32) -> bool {
  var neighborhood: array<u32, 8>;
  var neighbors = 0u;
  var transitions = 0u;
  for (var side = 0u; side < 8u; side++) {
    neighborhood[side] = getNeighborValue(index, side);
    neighbors += neighborhood[side];
  }
  for (var side = 0u; side < 8u; side++) {
    transitions += u32(neighborhood[side] == 0u && neighborhood[(side + 1u) % 8u] != 0u);
  }
  if (neighbors < 2u || neighbors > 6u || transitions != 1u) { return false; }
  if (phase == 0u) {
    return neighborhood[0] * neighborhood[2] * neighborhood[4] == 0u
      && neighborhood[2] * neighborhood[4] * neighborhood[6] == 0u;
  }
  return neighborhood[0] * neighborhood[2] * neighborhood[6] == 0u
    && neighborhood[0] * neighborhood[4] * neighborhood[6] == 0u;
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation,
        invocationCount: count,
        bindings: [
          {name: 'input', view: props.input, type: 'u32', access: 'read'},
          {name: 'first', view: first, type: 'u32', access: 'read_write'}
        ],
        body: 'first[firstOffset+index]=u32(input[inputOffset+index]!=0u);'
      }),
      createRasterIterationResetNode<Parameters>(graph, {id: `${id}-reset`, operation, state})
    ];
    // One extra iteration tests whether the final skeleton is a fixed point. Each iteration is
    // two dependent sub-iterations, so the loop gates per iteration, not per group of rounds.
    for (let iteration = 0; iteration < roundCount; iteration++) {
      for (let phase = 0; phase < 2; phase++) {
        const nodeId = `${id}-${iteration}-${phase}`;
        const {condition, extraResources} = getRasterIterationCondition<Parameters>(state, nodeId);
        nodes.push(
          createWGSLKernelNode(graph, {
            id: nodeId,
            operation,
            variant: `thinning-phase-${phase}`,
            invocationCount: count,
            bindings: [
              {name: 'previous', view: phase === 0 ? first : second, type: 'u32', access: 'read'},
              {name: 'next', view: phase === 0 ? second : first, type: 'u32', access: 'read_write'},
              {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'}
            ],
            declarations,
            body: `let original = previous[previousOffset + index];
  let remove = original != 0u && shouldRemove(index, ${phase}u);
  next[nextOffset + index] = select(original, 0u, remove);
  if (remove) { atomicOr(&status[statusOffset], 1u); }`,
            condition,
            extraResources
          })
        );
      }
      nodes.push(
        createRasterIterationGateNode<Parameters>(graph, {
          id: `${id}-gate-${iteration}`,
          operation,
          state,
          maxIterations: roundCount
        })
      );
    }
    const statusBindings = [
      {name: 'status', view: state.status, type: 'u32' as const, access: 'read' as const},
      {
        name: 'converged',
        view: output.status.converged,
        type: 'u32' as const,
        access: 'read_write' as const
      },
      ...(output.status.iterationLimitReached
        ? [
            {
              name: 'iterationLimitReached',
              view: output.status.iterationLimitReached,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : []),
      ...(output.status.iterationCount
        ? [
            {
              name: 'iterationCount',
              view: output.status.iterationCount,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-status`,
        operation,
        variant: 'iteration-status',
        invocationCount: 1,
        bindings: statusBindings,
        body: `converged[convergedOffset] = status[statusOffset + 3u];
  ${output.status.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = 1u - status[statusOffset + 3u];' : ''}
  ${output.status.iterationCount ? 'iterationCount[iterationCountOffset] = status[statusOffset + 2u];' : ''}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation,
        invocationCount: count,
        bindings: [
          {name: 'first', view: first, type: 'u32', access: 'read'},
          {name: 'skeleton', view: output.skeleton, type: 'u32', access: 'read_write'}
        ],
        // The current skeleton is always published; status.converged says whether thinning finished.
        body: 'skeleton[skeletonOffset+index]=first[firstOffset+index];'
      })
    );
    return nodes;
  }
}

export type GPURasterMedialAxisProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GraphDataView<'uint32'>;
  output: {medialAxis: GraphDataView<'uint32'>; radius: GraphDataView<'float32'>};
};
/**
 * Discrete maximal-inscribed-disk centers and their Euclidean radii in pixel units. All exterior
 * pixels are background. A disk center is removed only when its disk is contained in another
 * foreground disk; even-width plateaus can retain several centers. This is a maximal-disk medial
 * axis, distinct from topology-preserving one-pixel thinning. An exact banded distance transform
 * finds the nearest background center, then an integer primitive-lattice-step test classifies
 * each disk independently. Storage is linear in the number of cells; each dimension is at most
 * 32768 and the distance-transform scratch must fit the device's storage binding limits.
 *
 * Containment uses exact integer squared distances, correcting the previous all-pairs float32
 * comparison at tangent disks. Radii are correctly rounded float32 values in pixel units;
 * the grid affine does not rescale them. No thinning or approximate distance transform is used.
 */
export class GPURasterMedialAxis implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterMedialAxisProps;
  constructor(props: GPURasterMedialAxisProps) {
    this.id = props.id ?? 'raster-medial-axis';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.input],
      getGraphOutputViews(props.output)
    );
    // Squared pixel offsets, including the virtual exterior, must fit uint32 exactly.
    validateMaximumCount(
      this.id,
      'grid.width',
      props.grid.width,
      GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION
    );
    validateMaximumCount(
      this.id,
      'grid.height',
      props.grid.height,
      GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION
    );
    requireRasterLength(props.input, count, this.id, 'input');
    requireRasterLength(props.output.medialAxis, count, this.id, 'output.medialAxis');
    requireRasterLength(props.output.radius, count, this.id, 'output.radius');
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateRasterExpansionGraph(id, graph, [props.input, ...getGraphOutputViews(props.output)]);
    return getRasterMedialAxisNodes(graph, id, props);
  }
}
