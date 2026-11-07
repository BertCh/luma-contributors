// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from './terrain-flow-tracing';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from '../../gpu-raster/raster-expansion-utils';

export type GPUTerrainFlowDistanceProps = {
  id?: string;
  grid: GPURasterMetadata;
  flowDirections: GraphDataView<'uint32'>;
  /** Optional nonzero stream cells stop tracing before their outgoing transition. */
  streams?: GraphDataView<'uint32'>;
  /** Cost of each cell's outgoing edge, in explicit caller units. Otherwise planar affine length. */
  transitionCosts?: GraphDataView<'float32'>;
  /** Bounds each path walk. Default cell count; cycles and truncated paths remain unresolved. */
  maximumSteps?: number;
  output: {
    distance: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /** Terminal cell, 0xffffffff invalid, 0xfffffffe unresolved. */
    terminalCells: GraphDataView<'uint32'>;
  };
};

/** Downstream D8 path length, preserving invalid directions and bounded-search failures. */
export class GPUTerrainFlowDistance implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTerrainFlowDistanceProps;
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
    assert(Number.isInteger(props.maximumSteps ?? count) && (props.maximumSteps ?? count) > 0);
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
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUTerrainFlowDistance',
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
          {name: 'distance', view: output.distance, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'},
          {name: 'terminalCells', view: output.terminalCells, type: 'u32', access: 'read_write'}
        ],
        declarations: `${getTerrainFlowTracingWGSL(grid.width, grid.height)}\n${IS_FINITE_VALUE_WGSL}`,
        body: `var current = index; var total = 0.0; var terminal = UNRESOLVED_DIRECTION; var valid = false;
if (directions[directionsOffset+index] == NO_CELL) { terminal = NO_CELL; } else {
 for (var step = 0u; step < ${props.maximumSteps ?? count}u; step++) {
  let code = directions[directionsOffset+current];
  if (code != 0u && getFlowReceiver(current, code) == NO_CELL) { break; }
  ${props.streams ? 'if (streams[streamsOffset+current] == NO_CELL || streams[streamsOffset+current] == UNRESOLVED_DIRECTION) { break; } if (streams[streamsOffset+current] != 0u) { terminal = current; valid = true; break; }' : ''}
  let receiver = getFlowReceiver(current,code);
  if (receiver == NO_CELL) { terminal = current; valid = true; break; }
  ${
    props.transitionCosts
      ? 'let cost = costs[costsOffset+current];'
      : `let delta = vec2<f32>(f32(receiver % GRID_WIDTH) - f32(current % GRID_WIDTH), f32(receiver / GRID_WIDTH) - f32(current / GRID_WIDTH));
  let cost = length(vec2<f32>(${grid.affine[0]} * delta.x + ${grid.affine[1]} * delta.y, ${grid.affine[3]} * delta.x + ${grid.affine[4]} * delta.y));`
  }
  if (!isFiniteValue(cost) || cost < 0.0) { break; }
  total += cost; current = receiver;
 }
}
if (!isFiniteValue(total) && terminal != NO_CELL) { terminal = UNRESOLVED_DIRECTION; }
valid = valid && isFiniteValue(total);
distance[distanceOffset+index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), total, valid);
validity[validityOffset+index] = u32(valid); terminalCells[terminalCellsOffset+index] = terminal;`
      })
    ];
  }
}
