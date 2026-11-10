// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {checkTerrainViews, checkTerrainGraph} from './terrain-processing-utils';

import {createTerrainResidentTileNodes} from './terrain-tiled-resident';
import {validateRasterIterations} from '../../raster-analysis/cost-distance/raster-relaxation';

export type GPUTerrainFlowTile = {
  /** Global origin of the input buffer, including any halo. */
  column: number;
  row: number;
  width: number;
  height: number;
  /** Local owned rectangle [column, row, width, height]. Halo rows never inject runoff. */
  owned: readonly [number, number, number, number];
  /** D8 directions. In branching mode this compatibility view is ignored; use zero placeholders. */
  flowDirections: GraphDataView<'uint32'>;
  /** Eight fractions per cell, east first clockwise. Required only with routing: 'branching'. */
  fractions?: GraphDataView<'float32'>;
  runoff: GraphDataView<'float32'>;
  /** Resolved boolean stream mask (0/1); unresolved sentinels invalidate owned cells. */
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
  /** Disjoint owned rectangles must cover the entire domain. */
  tiles: readonly GPUTerrainFlowTile[];
  /** GPU DAG sweeps, default clamps width + height to [128, 1024]; each sweep follows bounded successor walks. */
  maximumIterations?: number;
};

/**
 * Owned-cell GPU exchange into a resident global DAG, with atomic transport and pointer-jumped
 * reach identity. Accumulation is in runoff units: it is a transport sum, so no length or area
 * unit and no `cellSizeMode` applies. NaN runoff marks a cell invalid.
 */
export class GPUTerrainTiledFlow implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainTiledFlowProps) {
    this.id = props.id ?? 'terrain-tiled-flow';
    validateRasterGrid(this.id, props.grid);
    validateRasterIterations(this.id, 'maximumIterations', props.maximumIterations ?? 128);
    let ownedCount = 0;
    const rectangles: number[][] = [];
    const inputs: GraphDataView[] = [],
      outputs: GraphDataView[] = [];
    for (const [tileIndex, tile] of props.tiles.entries()) {
      const name = `tiles[${tileIndex}]`;
      const [column, row, width, height] = tile.owned;
      validateRequirement(
        this.id,
        [tile.column, tile.row, tile.width, tile.height, ...tile.owned].every(Number.isInteger),
        `${name} column, row, width, height and owned must be integers`
      );
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
    checkTerrainViews(this.id, inputs, outputs);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    checkTerrainGraph(
      id,
      graph,
      props.tiles.flatMap(tile => [
        tile.flowDirections,
        tile.runoff,
        tile.streams,
        ...(tile.fractions ? [tile.fractions] : []),
        ...Object.values(tile.output)
      ])
    );
    return createTerrainResidentTileNodes(graph, props, id);
  }
}
