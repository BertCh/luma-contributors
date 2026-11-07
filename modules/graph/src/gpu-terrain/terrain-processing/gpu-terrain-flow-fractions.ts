// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {
  validateRasterGrid,
  validateRasterGridAxisAligned
} from '../../gpu-raster/raster-band-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {getRasterGridWGSL} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  getFlowFractionWGSL,
  type TerrainFlowMultipleRouting
} from '../hydrology/terrain-flow-routing';
import {checkTerrainViews, checkTerrainGraph, terrainBinding} from './terrain-processing-utils';

export type GPUTerrainFlowFractionsProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Same canonical routing surface used by GPUTerrainFlow, including fill when enabled. */
  elevation: GraphDataView<'float32'>;
  /** Existing GPUTerrainFlow D8 fallback directions, preserving unresolved sentinels. */
  flowDirections: GraphDataView<'uint32'>;
  /** Existing GPUTerrainFlow parameters (including MFD exponent). */
  parameters: GraphDataView<'float32'>;
  routing: TerrainFlowMultipleRouting;
  cellSizeMode?: GPURasterCellSizeMode;
  output: {
    fractions: GraphDataView<'float32'>;
    /** Ground lengths using the same row-dependent metric as flow routing. */
    edgeLengths: GraphDataView<'float32'>;
  };
};

/** Materializes the existing D-infinity/MFD routing kernel for branching consumers, without duplicating routing mathematics. */
export class GPUTerrainFlowFractions implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainFlowFractionsProps) {
    this.id = props.id ?? 'terrain-flow-fractions';
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    assert(
      props.elevation.length === count &&
        props.flowDirections.length === count &&
        props.parameters.length >= 8
    );
    assert(
      props.output.fractions.length === count * 8 && props.output.edgeLengths.length === count * 8
    );
    checkTerrainViews(
      this.id,
      [props.elevation, props.flowDirections, props.parameters],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.elevation.length;
    checkTerrainGraph(id, graph, [
      props.elevation,
      props.flowDirections,
      props.parameters,
      ...Object.values(props.output)
    ]);
    const receivers = createTransientView(graph, `${id}-receivers`, 'uint32', count);
    const routing = createTransientView(graph, `${id}-routing`, 'uint32', count * 2);
    const grid = getRasterGridWGSL({
      width: props.grid.width,
      height: props.grid.height,
      cellSizeMode: props.cellSizeMode ?? 'uniform'
    });
    const inputs = [
      terrainBinding('surface', props.elevation),
      terrainBinding('receivers', receivers),
      terrainBinding('settings', props.parameters)
    ];
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-receivers`,
        operation: 'GPUTerrainFlowFractions',
        invocationCount: count,
        declarations: grid,
        bindings: [
          terrainBinding('directions', props.flowDirections),
          terrainBinding('settings', props.parameters),
          terrainBinding('receivers', receivers, 'read_write')
        ],
        body: `let code = directions[directionsOffset + index]; var receiver = GRID_NONE;
if (code >= 1u && code <= 128u && (code & (code - 1u)) == 0u) { receiver = getD8Neighbor(index, firstTrailingBit(code)); }
receivers[receiversOffset + index] = receiver;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPUTerrainFlowFractions',
        invocationCount: count,
        declarations: `${grid}\nconst ROUTING_INVALID: u32 = 0xffffffffu;\n${getFlowFractionWGSL(props.routing, true)}`,
        bindings: [...inputs, terrainBinding('routing', routing, 'read_write')],
        body: 'computeRouting(index);'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUTerrainFlowFractions',
        invocationCount: count,
        declarations: `${grid}\nconst ROUTING_INVALID: u32 = 0xffffffffu;\n${getFlowFractionWGSL(props.routing, false)}`,
        bindings: [
          ...inputs,
          terrainBinding('routing', routing),
          terrainBinding('directions', props.flowDirections),
          terrainBinding('fractions', props.output.fractions, 'read_write'),
          terrainBinding('lengths', props.output.edgeLengths, 'read_write')
        ],
        body: `let code = directions[directionsOffset + index];
let valid = isFiniteValue(surface[surfaceOffset + index]) && (code == 0u || (code <= 128u && (code & (code - 1u)) == 0u && receivers[receiversOffset + index] != GRID_NONE));
for (var direction = 0u; direction < 8u; direction++) {
 fractions[fractionsOffset + index * 8u + direction] = select(getQuietNaN(), getFlowFraction(index, direction), valid);
 lengths[lengthsOffset + index * 8u + direction] = getD8Distance(direction, index / GRID_WIDTH);
}`
      })
    ];
  }
}
