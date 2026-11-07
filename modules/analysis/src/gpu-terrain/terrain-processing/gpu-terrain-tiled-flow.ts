// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
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

/** Owned-cell GPU exchange into a resident global DAG, with atomic transport and pointer-jumped reach identity. */
export class GPUTerrainTiledFlow implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainTiledFlowProps) {
    this.id = props.id ?? 'terrain-tiled-flow';
    validateRasterGrid(this.id, props.grid);
    assert(
      Number.isInteger(props.grid.width) &&
        props.grid.width > 0 &&
        Number.isInteger(props.grid.height) &&
        props.grid.height > 0
    );
    validateRasterIterations(this.id, 'maximumIterations', props.maximumIterations ?? 128);
    let ownedCount = 0;
    const rectangles: number[][] = [];
    const inputs: GraphDataView[] = [],
      outputs: GraphDataView[] = [];
    for (const tile of props.tiles) {
      const [column, row, width, height] = tile.owned;
      assert(
        [tile.column, tile.row, tile.width, tile.height, ...tile.owned].every(Number.isInteger)
      );
      assert(
        tile.width > 0 &&
          tile.height > 0 &&
          column >= 0 &&
          row >= 0 &&
          width > 0 &&
          height > 0 &&
          column + width <= tile.width &&
          row + height <= tile.height
      );
      const rectangle = [tile.column + column, tile.row + row, width, height];
      assert(
        rectangle[0] >= 0 &&
          rectangle[1] >= 0 &&
          rectangle[0] + width <= props.grid.width &&
          rectangle[1] + height <= props.grid.height
      );
      for (const other of rectangles)
        assert(
          rectangle[0] >= other[0] + other[2] ||
            other[0] >= rectangle[0] + width ||
            rectangle[1] >= other[1] + other[3] ||
            other[1] >= rectangle[1] + height
        );
      rectangles.push(rectangle);
      ownedCount += width * height;
      if (props.routing === 'branching') {
        assert(tile.fractions && tile.fractions.length === tile.width * tile.height * 8);
        inputs.push(tile.fractions);
      } else {
        assert(!tile.fractions);
      }
      inputs.push(tile.flowDirections, tile.runoff, tile.streams);
      outputs.push(...Object.values(tile.output));
      for (const view of [
        tile.flowDirections,
        tile.runoff,
        tile.streams,
        ...Object.values(tile.output)
      ])
        assert(view.length === tile.width * tile.height);
    }
    assert(ownedCount === props.grid.width * props.grid.height);
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
