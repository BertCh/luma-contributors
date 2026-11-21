// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import type {Device} from '@luma.gl/core';
import {snapshotRasterGrid} from '../../raster-analysis/raster-grid/raster-construction-snapshot';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  validateTerrainFlowDirectionEncoding,
  type GPUTerrainFlowDirectionEncoding
} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  validateTerrainStatusPorts
} from './terrain-processing-utils';

import {createTerrainResidentTileNodes} from './terrain-tiled-resident';
import {validateRasterIterations} from '../../raster-analysis/cost-distance/raster-relaxation';
import {validateTerrainBranchingFlowScratch} from './terrain-branching-flow-budget';

import {
  getTerrainResolutionStatusViews,
  type TerrainResolutionStatusPort
} from './terrain-resolution-status';
import {
  planTerrainD8Tiles,
  getTerrainTiledFlowExecutionPlan,
  TERRAIN_TILED_MAXIMUM_CELL_COUNT,
  TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT,
  validateTerrainD8TileBindings,
  type TerrainD8TilePlan
} from './terrain-tiled-plan';
import {createTerrainBoundaryTileNodes} from './terrain-tiled-boundary';

export {
  getTerrainTiledFlowExecutionPlan,
  TERRAIN_TILED_MAXIMUM_CELL_COUNT,
  TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT,
  TERRAIN_TILED_MAXIMUM_FOREST_CELL_COUNT,
  TERRAIN_TILED_MAXIMUM_GRID_EXTENT,
  TERRAIN_TILED_MAXIMUM_INDEX_WORD_COUNT
} from './terrain-tiled-plan';

export type GPUTerrainFlowTile = {
  /** Global origin of the input buffer, including any halo. */
  column: number;
  row: number;
  width: number;
  height: number;
  /** Local owned rectangle [column, row, width, height]. Halo rows never inject runoff. */
  owned: readonly [number, number, number, number];
  /**
   * D8 directions (ESRI codes). In branching mode the routing comes from `fractions` and only one
   * word is read from this view: `GPU_TERRAIN_FLOW_NONE` (0xffffffff, a cell without data) makes
   * the cell invalid (status 0, NaN accumulation), as it does in D8 mode. Pass the
   * `flowDirections` of `GPUTerrainFlow`, or zero placeholders when every cell has data.
   */
  flowDirections: GraphDataView<'uint32'>;
  /** Eight fractions per cell, east first clockwise. Required only with routing: 'branching'. */
  fractions?: GraphDataView<'float32'>;
  runoff: GraphDataView<'float32'>;
  /**
   * Stream mask: nonzero is a stream cell (so a mask of Strahler orders works), 0 is not. The
   * unresolved sentinels 0xffffffff and 0xfffffffe invalidate owned cells.
   */
  streams: GraphDataView<'uint32'>;
  output: {
    accumulation: GraphDataView<'float32'>;
    /** 0 halo/invalid, 1 resolved, 2 cycle, invalid dependency or exhausted budget. */
    status: GraphDataView<'uint32'>;
    /** Global source cell of each stream chain. Branch receivers and confluences start new IDs; 0xffffffff for nonstreams/unresolved. */
    reachIds: GraphDataView<'uint32'>;
  };
};
export type GPUTerrainTiledFlowProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Routing is globally prepared. Branching consumes D-infinity/MFD fractions without rerouting at seams. */
  routing?: 'd8' | 'branching';
  /**
   * `'auto'` (default) retains resident transport while its fraction plane fits, and otherwise
   * uses bounded D8 tile scratch plus a coarse boundary forest. `'resident'` keeps the existing
   * reference route; `'tiled'` explicitly selects the D8 boundary route. Branching is resident.
   * The tiled route uses exact 320-bit sums, rounded at boundary summaries and output.
   * It preserves exact integer sums through 2^24 and agrees with the CPU sum within 2e-5 relative
   * on ordinary finite inputs. Long resident float32 chains can accumulate larger rounding error.
   */
  execution?: 'auto' | 'resident' | 'tiled';
  /**
   * Numbering of every tile's `flowDirections`: `'esri'` (default), `'whitebox'` (the
   * WhiteboxTools default pointer) or `'taudem'`, see `GPUTerrainFlowDirectionEncoding`.
   * Compile-time.
   */
  directionEncoding?: GPUTerrainFlowDirectionEncoding;
  /** Disjoint owned rectangles must cover the entire domain. */
  tiles: readonly GPUTerrainFlowTile[];
  /** Resident GPU DAG sweeps, default clamps width + height to [128, 1024]. Tiled D8 uses bounded logarithmic forest passes and ignores this sweep budget. */
  maximumIterations?: number;
  /** Optional aggregate diagnostics over global owned cells, never halo copies. */
  output?: {
    /**
     * `invalidCount` counts invalid owned cells (state 0). `converged` is 1 when no owned cell
     * remains unresolved (state 2). `iterationLimitReached` is 1 only when the solver exhausts
     * maximumIterations before its no-progress stop. A stalled cycle can report both flags 0.
     * An entirely invalid domain converges with its invalidCount.
     */
    status: TerrainResolutionStatusPort;
  };
};

/**
 * Owned-cell GPU exchange into a resident global DAG, with atomic transport and pointer-jumped
 * reach identity. Accumulation is in runoff units: it is a transport sum, so no length or area
 * unit and no `cellSizeMode` applies. NaN runoff marks a cell invalid.
 * Tiled D8 schedules one tile at a time, retaining only maximum-tile scratch and a graph of owned
 * boundary cells. Caller input/output views remain caller-owned; no host storage paging or terrain
 * filling is performed. All tile geometry, lengths and offsets use dispatch parameters.
 * Resident routing accepts at most 536,870,911 global cells; tiled D8 accepts at most
 * 4,294,967,295 global cells, with each axis at most 2,147,483,646. Each owned tile and the
 * combined boundary forest contain at most 536,870,911 cells; tile index arrays contain at most
 * 4,294,967,295 scalar uint32 words. Tile and boundary tour-sum planes use 32 bytes per forest cell.
 * Each scratch binding must fit maxStorageBufferBindingSize and maxBufferSize on the device;
 * the selected route's command-node estimate must be at most 32,768. A 32768 x 16384 grid is tiled.
 */
export class GPUTerrainTiledFlow implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  /** Estimate for the selected route on the views' graph device. */
  readonly selectedExecution: 'resident' | 'tiled';
  /** Conservative cross-device command bound, separate from selected-route admission. */
  readonly worstCaseCommandNodeCount: number;
  readonly props: GPUTerrainTiledFlowProps;
  private readonly tiledPlan?: TerrainD8TilePlan;
  constructor(inputProps: GPUTerrainTiledFlowProps) {
    const captured = snapshotConstructionFields(inputProps, [
      'id',
      'grid',
      'routing',
      'execution',
      'directionEncoding',
      'tiles',
      'maximumIterations',
      'output'
    ]);
    const props: GPUTerrainTiledFlowProps = {
      ...captured,
      grid: snapshotRasterGrid(captured.grid),
      tiles: captured.tiles.map(inputTile => {
        const tile = snapshotConstructionFields(inputTile, [
          'column',
          'row',
          'width',
          'height',
          'owned',
          'flowDirections',
          'fractions',
          'runoff',
          'streams',
          'output'
        ]);
        return {
          ...tile,
          owned: [...tile.owned] as [number, number, number, number],
          output: snapshotConstructionFields(tile.output, ['accumulation', 'status', 'reachIds'])
        };
      }),
      ...(captured.output
        ? {
            output: {
              status: snapshotConstructionFields(captured.output.status, [
                'converged',
                'iterationLimitReached',
                'invalidCount'
              ])
            }
          }
        : {})
    };
    this.props = props;
    this.id = props.id ?? 'terrain-tiled-flow';
    const count = validateRasterGrid(this.id, props.grid);
    if (props.execution === 'resident' || props.routing === 'branching') {
      validateCapacity(
        this.id,
        'resident grid cell count',
        count,
        1,
        TERRAIN_TILED_MAXIMUM_CELL_COUNT
      );
    } else {
      validateCapacity(
        this.id,
        'D8 grid cell count',
        count,
        1,
        TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT
      );
    }
    validateRequirement(
      this.id,
      props.execution === undefined || ['auto', 'resident', 'tiled'].includes(props.execution),
      "execution must be 'auto', 'resident' or 'tiled'"
    );
    validateRequirement(
      this.id,
      props.execution !== 'tiled' || props.routing !== 'branching',
      "execution 'tiled' requires routing 'd8'"
    );
    validateRequirement(
      this.id,
      props.routing === undefined || props.routing === 'd8' || props.routing === 'branching',
      "routing must be 'd8' or 'branching'"
    );
    validateTerrainFlowDirectionEncoding(this.id, props.directionEncoding);
    const maximumIterations =
      props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024);
    validateRasterIterations(this.id, 'maximumIterations', maximumIterations);
    validateRequirement(this.id, props.tiles.length > 0, 'tiles must contain at least one tile');
    const executionPlan = this.getExecutionPlan(
      (props.tiles[0].streams.buffer.graph as GPUCommandGraph).device
    );
    this.selectedExecution = executionPlan.selectedExecution;
    this.estimatedCommandNodeCount = executionPlan.estimatedCommandNodeCount;
    this.worstCaseCommandNodeCount = executionPlan.worstCaseCommandNodeCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateTerrainStatusPorts(this.id, {...props.output?.status});
    let ownedCount = 0;
    const rectangles: number[][] = [];
    const inputs: GraphDataView[] = [],
      outputs: GraphDataView[] = [];
    for (const [tileIndex, tile] of props.tiles.entries()) {
      const name = `tiles[${tileIndex}]`;
      const [column, row, width, height] = tile.owned;
      validateRequirement(
        this.id,
        [tile.column, tile.row, tile.width, tile.height, ...tile.owned].every(Number.isSafeInteger),
        `${name} column, row, width, height and owned must be integers`
      );
      validateCapacity(this.id, `${name} cell count`, tile.width * tile.height);
      validateRequirement(
        this.id,
        tile.width > 0 &&
          tile.height > 0 &&
          column >= 0 &&
          row >= 0 &&
          width > 0 &&
          height > 0 &&
          column + width <= tile.width &&
          row + height <= tile.height,
        `${name} owned rectangle must be non-empty and inside the tile`
      );
      const rectangle = [tile.column + column, tile.row + row, width, height];
      validateRequirement(
        this.id,
        rectangle[0] >= 0 &&
          rectangle[1] >= 0 &&
          rectangle[0] + width <= props.grid.width &&
          rectangle[1] + height <= props.grid.height,
        `${name} owned rectangle must be inside the grid`
      );
      for (const other of rectangles)
        validateRequirement(
          this.id,
          rectangle[0] >= other[0] + other[2] ||
            other[0] >= rectangle[0] + width ||
            rectangle[1] >= other[1] + other[3] ||
            other[1] >= rectangle[1] + height,
          `${name} owned rectangle must not overlap another tile's`
        );
      rectangles.push(rectangle);
      ownedCount += width * height;
      if (props.routing === 'branching') {
        validateCapacity(
          this.id,
          `${name} cell count with routing 'branching'`,
          tile.width * tile.height,
          1,
          TERRAIN_TILED_MAXIMUM_CELL_COUNT
        );
        validateRequirement(
          this.id,
          Boolean(tile.fractions),
          `${name} fractions are required with routing 'branching'`
        );
        validateViewLength(
          this.id,
          `${name} fractions`,
          tile.fractions!,
          tile.width * tile.height * 8
        );
        inputs.push(tile.fractions!);
      } else {
        validateRequirement(this.id, !tile.fractions, `${name} fractions need routing 'branching'`);
      }
      inputs.push(tile.flowDirections, tile.runoff, tile.streams);
      outputs.push(...Object.values(tile.output));
      for (const [viewName, view] of [
        ['flowDirections', tile.flowDirections],
        ['runoff', tile.runoff],
        ['streams', tile.streams],
        ...Object.entries(tile.output).map(
          ([outputName, output]) => [`output.${outputName}`, output] as const
        )
      ] as const)
        validateViewLength(this.id, `${name} ${viewName}`, view, tile.width * tile.height);
    }
    validateRequirement(
      this.id,
      ownedCount === props.grid.width * props.grid.height,
      'owned rectangles must cover every grid cell exactly once'
    );
    const allOutputs = [...outputs, ...getTerrainResolutionStatusViews(props.output?.status)];
    const publicationReadsStreams = this.selectedExecution === 'tiled';
    checkTerrainViews(this.id, inputs, allOutputs, [
      ...props.tiles.map(tile => ({
        reads: publicationReadsStreams ? [tile.streams] : [],
        writes: Object.values(tile.output)
      })),
      ...getTerrainResolutionStatusViews(props.output?.status).map(view => ({writes: [view]}))
    ]);
    if (this.selectedExecution === 'tiled') {
      this.tiledPlan = planTerrainD8Tiles(this.id, props);
    }
  }
  /** The same pure selector used for admission and graph declaration. */
  getExecutionPlan(device: Pick<Device, 'limits'>) {
    return getTerrainTiledFlowExecutionPlan(this.props, device);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    checkTerrainGraph(id, graph, [
      ...props.tiles.flatMap(tile => [
        tile.flowDirections,
        tile.runoff,
        tile.streams,
        ...(tile.fractions ? [tile.fractions] : []),
        ...Object.values(tile.output)
      ]),
      ...getTerrainResolutionStatusViews(props.output?.status)
    ]);
    const count = props.grid.width * props.grid.height;
    const useTiles = this.getExecutionPlan(graph.device).selectedExecution === 'tiled';
    if (useTiles) {
      validateTerrainD8TileBindings(graph, id, this.tiledPlan!);
      validateGraphChunkGroupKernelSupport(graph, id, 8);
      return createTerrainBoundaryTileNodes(graph, props, id, this.tiledPlan!);
    }
    // The fraction plane dominates every cell-proportional transient in the resident composition.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'resident fractions scratch',
      count * 8 * 4,
      'reduce grid dimensions'
    );
    validateTerrainBranchingFlowScratch(
      graph,
      id,
      count,
      props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024)
    );
    validateGraphChunkGroupKernelSupport(graph, id, props.routing === 'branching' ? 7 : 6);
    return createTerrainResidentTileNodes(graph, props, id);
  }
}
