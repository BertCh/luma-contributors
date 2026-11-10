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
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../raster-analysis/raster-band-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {getRasterGridWGSL} from '../../raster-analysis/cost-distance/raster-grid-utils';
import {validateViewLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getFlowFractionWGSL,
  type TerrainFlowMultipleRouting
} from '../hydrology/terrain-flow-routing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  createTerrainValidityMask,
  terrainBinding,
  validateTerrainValidityView
} from './terrain-processing-utils';

export type GPUTerrainFlowFractionsProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Same canonical routing surface used by GPUTerrainFlow, including fill when enabled. */
  elevation: GraphDataView<'float32'>;
  /** Optional per-cell validity of `elevation` (uint32, nonzero valid); a zero word counts as NaN, so the cell has NaN fractions. */
  validity?: GraphDataView<'uint32'>;
  /** Existing GPUTerrainFlow D8 fallback directions, preserving unresolved sentinels. */
  flowDirections: GraphDataView<'uint32'>;
  /** Existing GPUTerrainFlow parameters (including MFD exponent). */
  parameters: GraphDataView<'float32'>;
  routing: TerrainFlowMultipleRouting;
  /** Ground model of the edge lengths and of the `parameters` cell size. Default `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  output: {
    fractions: GraphDataView<'float32'>;
    /** Eight lengths per cell in the units of the `parameters` cell size (ground meters under `cellSizeMode`), using the same row-dependent metric as flow routing. */
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
    validateViewLength(this.id, 'elevation', props.elevation, count);
    validateViewLength(this.id, 'flowDirections', props.flowDirections, count);
    validateTerrainValidityView(this.id, props.validity, count);
    validateRasterParameters(this.id, props.parameters, 8);
    validateViewLength(this.id, 'output.fractions', props.output.fractions, count * 8);
    validateViewLength(this.id, 'output.edgeLengths', props.output.edgeLengths, count * 8);
    checkTerrainViews(
      this.id,
      [
        props.elevation,
        props.flowDirections,
        props.parameters,
        ...(props.validity ? [props.validity] : [])
      ],
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
      ...(props.validity ? [props.validity] : []),
      ...Object.values(props.output)
    ]);
    const masked = createTerrainValidityMask(graph, {
      id,
      operation: 'GPUTerrainFlowFractions',
      values: props.elevation,
      validity: props.validity
    });
    const receivers = createTransientView(graph, `${id}-receivers`, 'uint32', count);
    const routing = createTransientView(graph, `${id}-routing`, 'uint32', count * 2);
    const grid = getRasterGridWGSL({
      width: props.grid.width,
      height: props.grid.height,
      cellSizeMode: props.cellSizeMode ?? 'uniform'
    });
    const inputs = [
      terrainBinding('surface', masked.values),
      terrainBinding('receivers', receivers),
      terrainBinding('settings', props.parameters)
    ];
    return [
      ...masked.nodes,
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
