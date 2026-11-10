// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from './terrain-flow-tracing';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateRasterGridAxisAligned} from '../../raster-analysis/raster-band-utils';
import {
  createRasterJacobiLoopNodes,
  getRasterJacobiLoopNodeCount,
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from '../../raster-analysis/raster-expansion-utils';
import {
  getTerrainGroundMeasureWGSL,
  getTerrainJumpRoundCount
} from '../terrain-processing/terrain-processing-utils';

export type GPUTerrainFlowDistanceProps = {
  id?: string;
  grid: GPURasterMetadata;
  flowDirections: GraphDataView<'uint32'>;
  /** Optional nonzero stream cells stop tracing before their outgoing transition. */
  streams?: GraphDataView<'uint32'>;
  /**
   * Cost of each cell's outgoing edge, in explicit caller units. Otherwise the length of the
   * edge: affine units by default (geographic degrees are not ground meters), or ground meters
   * under `cellSizeMode`. `output.distance` is in the same units.
   */
  transitionCosts?: GraphDataView<'float32'>;
  /** Ground model for the default edge length. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Bounds each path, counted in cells including the terminal one. Default cell count; cycles
   * and truncated paths remain unresolved. Compile-time: the pointer-jumping round count is
   * `ceil(log2(min(maximumSteps, cell count)))`.
   */
  maximumSteps?: number;
  output: {
    distance: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /** Terminal cell, 0xffffffff invalid, 0xfffffffe unresolved. */
    terminalCells: GraphDataView<'uint32'>;
  };
};

/**
 * Downstream D8 path length, preserving invalid directions and bounded-search failures. Every
 * cell's path is resolved by pointer jumping with accumulated distance: `ceil(log2(path bound))`
 * gated rounds, each doubling the jump, instead of one thread walking the whole path. Distances
 * are sums in a different order than a serial walk, so they agree to float32 rounding.
 */
export class GPUTerrainFlowDistance implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTerrainFlowDistanceProps;
  /** Number of command nodes `getCommandNodes` returns; bounded by the command node budget. */
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPUTerrainFlowDistanceProps) {
    this.id = props.id ?? 'terrain-flow-distance';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.flowDirections, props.streams, props.transitionCosts],
      Object.values(props.output)
    );
    for (const view of [
      props.flowDirections,
      props.streams,
      props.transitionCosts,
      ...Object.values(props.output)
    ])
      requireRasterLength(view, count);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateCapacity(this.id, 'maximumSteps', props.maximumSteps ?? count);
    // Seed and output nodes plus the gated pointer-jumping loop.
    this.estimatedCommandNodeCount =
      2 +
      getRasterJacobiLoopNodeCount(
        getTerrainJumpRoundCount(Math.min(props.maximumSteps ?? count, count))
      );
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      props.flowDirections,
      props.streams,
      props.transitionCosts,
      ...Object.values(output)
    ]);
    const maximumSteps = props.maximumSteps ?? count;
    const rounds = getTerrainJumpRoundCount(Math.min(maximumSteps, count));
    // Per cell: next cell on the path (NO_CELL when done), distance, cells on the path so far, terminal cell.
    const states = [0, 1].map(side =>
      createTransientView(graph, `${id}-state-${side}`, 'uint32', count * 4)
    );
    const converged = createTransientView(graph, `${id}-converged`, 'uint32', 1);
    const declarations = `${getTerrainFlowTracingWGSL(grid.width, grid.height)}
${IS_FINITE_VALUE_WGSL}
${getTerrainGroundMeasureWGSL(grid, props.cellSizeMode)}`;
    const operation = 'GPUTerrainFlowDistance';
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-seed`,
        operation,
        invocationCount: count,
        bindings: [
          {name: 'directions', view: props.flowDirections, type: 'u32', access: 'read'},
          ...(props.streams
            ? [
                {
                  name: 'streams',
                  view: props.streams,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...(props.transitionCosts
            ? [
                {
                  name: 'costs',
                  view: props.transitionCosts,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'state', view: states[0], type: 'u32', access: 'read_write'}
        ],
        declarations,
        // kind 0 extends the path to the receiver, 1 ends it at this cell, 2 cannot be resolved
        // (invalid direction, unresolved stream word or unusable cost).
        body: `let code = directions[directionsOffset+index];
let receiver = getFlowReceiver(index, code);
var kind = 0u; var cost = 0.0;
if (code != 0u && receiver == NO_CELL) { kind = 2u; } else {
 ${props.streams ? 'let stream = streams[streamsOffset+index]; if (stream == NO_CELL || stream == UNRESOLVED_DIRECTION) { kind = 2u; } else if (stream != 0u) { kind = 1u; }' : ''}
 if (kind == 0u && receiver == NO_CELL) { kind = 1u; }
 if (kind == 0u) {
  cost = ${props.transitionCosts ? 'costs[costsOffset+index]' : 'getGroundEdgeLength(index, receiver)'};
  if (!isFiniteValue(cost) || cost < 0.0) { kind = 2u; cost = 0.0; }
 }
}
let offset = stateOffset + index * 4u;
state[offset] = select(NO_CELL, receiver, kind == 0u);
state[offset + 1u] = bitcast<u32>(cost);
state[offset + 2u] = 1u;
state[offset + 3u] = select(UNRESOLVED_DIRECTION, index, kind == 1u);`
      }),
      ...createRasterJacobiLoopNodes(graph, {
        id: `${id}-jump`,
        operation,
        invocationCount: count,
        roundCount: rounds,
        status: {converged},
        createRound: (
          {nodeId, iteration, slotBinding, markChangedWGSL, condition, extraResources},
          status
        ) =>
          createWGSLKernelNode(graph, {
            id: nodeId,
            operation,
            variant: 'jump',
            invocationCount: count,
            bindings: [
              {name: 'previous', view: states[iteration % 2], type: 'u32', access: 'read'},
              {name: 'next', view: states[(iteration + 1) % 2], type: 'u32', access: 'read_write'},
              {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'},
              slotBinding
            ],
            declarations: `const NO_CELL: u32 = 0xffffffffu;`,
            condition,
            extraResources,
            body: `let offset = previousOffset + index * 4u; let destination = nextOffset + index * 4u;
for (var component = 0u; component < 4u; component++) { next[destination + component] = previous[offset + component]; }
let parent = previous[offset];
if (parent != NO_CELL) {
 let ancestor = previousOffset + parent * 4u;
 next[destination] = previous[ancestor];
 next[destination + 1u] = bitcast<u32>(bitcast<f32>(previous[offset + 1u]) + bitcast<f32>(previous[ancestor + 1u]));
 next[destination + 2u] = min(${count + 1}u, previous[offset + 2u] + previous[ancestor + 2u]);
 next[destination + 3u] = previous[ancestor + 3u];
 ${markChangedWGSL}
}`
          })
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-output`,
        operation,
        invocationCount: count,
        bindings: [
          {name: 'directions', view: props.flowDirections, type: 'u32', access: 'read'},
          {name: 'state', view: states[rounds % 2], type: 'u32', access: 'read'},
          {name: 'distance', view: output.distance, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          {name: 'terminalCells', view: output.terminalCells, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `let offset = stateOffset + index * 4u;
let total = bitcast<f32>(state[offset + 1u]);
var terminal = state[offset + 3u];
let startsInvalid = directions[directionsOffset+index] == NO_CELL;
let valid = !startsInvalid && state[offset] == NO_CELL && terminal != UNRESOLVED_DIRECTION && state[offset + 2u] <= ${maximumSteps}u && isFiniteValue(total);
terminal = select(select(UNRESOLVED_DIRECTION, NO_CELL, startsInvalid), terminal, valid);
distance[distanceOffset+index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), total, valid);
validity[validityOffset+index] = u32(valid); terminalCells[terminalCellsOffset+index] = terminal;`
      })
    ];
  }
}
