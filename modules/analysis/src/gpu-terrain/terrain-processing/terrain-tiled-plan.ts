// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import {getTerrainBranchingFlowCommandNodeCount} from './terrain-branching-flow-budget';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import type {GPUTerrainTiledFlowProps} from './gpu-terrain-tiled-flow';

// Each wide-sum plane holds four uint32 words per tour edge (eight per forest cell).
export const TERRAIN_TILED_MAXIMUM_FOREST_CELL_COUNT = Math.floor(0xffffffff / 8);
// Index records and their scalar offsets are uint32-addressed.
export const TERRAIN_TILED_MAXIMUM_INDEX_WORD_COUNT = 0xffffffff;
// The published radix sort has no public command-node estimate; includes all its passes.
const TERRAIN_TILED_PRIMITIVE_NODE_ALLOWANCE = 512;
// Boundary extraction, summary, replay, publication and aggregate tally are one node per tile.
const TERRAIN_TILED_COMMAND_NODES_PER_TILE = 5;
// Neighbour coordinates use signed offsets of one, so both coordinate + 1 and the extent fit i32.
export const TERRAIN_TILED_MAXIMUM_GRID_EXTENT = 0x7ffffffe;
export const TERRAIN_TILED_MAXIMUM_CELL_COUNT = Math.floor(0xffffffff / 8);
export const TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT = 0xffffffff;
const TERRAIN_TILED_STATUS_NODE_ALLOWANCE = 6;

/** Pure route selection and command bounds; never reads terrain rows or changes a graph. */
export function getTerrainTiledFlowExecutionPlan(
  props: GPUTerrainTiledFlowProps,
  device: Pick<Device, 'limits'>
) {
  const id = props.id ?? 'terrain-tiled-flow';
  validateRequirement(
    id,
    ['auto', 'resident', 'tiled'].includes(props.execution ?? 'auto'),
    "execution must be 'auto', 'resident' or 'tiled'"
  );
  validateRequirement(
    id,
    ['d8', 'branching'].includes(props.routing ?? 'd8'),
    "routing must be 'd8' or 'branching'"
  );
  validateRequirement(
    id,
    props.execution !== 'tiled' || props.routing !== 'branching',
    "execution 'tiled' requires routing 'd8'"
  );
  const count = props.grid.width * props.grid.height;
  const maximumIterations =
    props.maximumIterations ?? Math.max(128, Math.min(1024, props.grid.width + props.grid.height));
  const selectedExecution: 'resident' | 'tiled' =
    props.routing !== 'branching' &&
    props.execution !== 'resident' &&
    (props.execution === 'tiled' ||
      count > TERRAIN_TILED_MAXIMUM_CELL_COUNT ||
      count * 32 > Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize))
      ? 'tiled'
      : 'resident';
  const residentCommandNodeCount =
    getTerrainBranchingFlowCommandNodeCount(maximumIterations) +
    props.tiles.length * 2 +
    1 +
    Math.ceil(Math.log2(count)) +
    TERRAIN_TILED_STATUS_NODE_ALLOWANCE;
  const tiledCommandNodeCount = getTerrainD8TileCommandNodeCount(props.tiles.length);
  return {
    requestedExecution: props.execution ?? 'auto',
    selectedExecution,
    residentFractionsByteLength: count * 32,
    estimatedCommandNodeCount:
      selectedExecution === 'tiled' ? tiledCommandNodeCount : residentCommandNodeCount,
    worstCaseCommandNodeCount:
      props.routing === 'branching' || props.execution === 'resident'
        ? residentCommandNodeCount
        : props.execution === 'tiled'
          ? tiledCommandNodeCount
          : Math.max(residentCommandNodeCount, tiledCommandNodeCount)
  };
}

/** Includes all published-sort passes and the fixed coarse-graph/status nodes. @internal */
export function getTerrainD8TileCommandNodeCount(tileCount: number): number {
  return tileCount * TERRAIN_TILED_COMMAND_NODES_PER_TILE + TERRAIN_TILED_PRIMITIVE_NODE_ALLOWANCE;
}

/** Metadata only; no raster rows are read on the CPU. @internal */
export function planTerrainD8Tiles(id: string, props: GPUTerrainTiledFlowProps) {
  validateMaximumCount(id, 'grid.width', props.grid.width, TERRAIN_TILED_MAXIMUM_GRID_EXTENT);
  validateMaximumCount(id, 'grid.height', props.grid.height, TERRAIN_TILED_MAXIMUM_GRID_EXTENT);
  let boundaryCount = 0;
  let maximumCellCount = 0;
  const rectangles: number[] = [];
  const tiles = props.tiles.map(tile => {
    const [column, row, width, height] = tile.owned;
    const count = width * height;
    const perimeter = width === 1 || height === 1 ? count : width * 2 + (height - 2) * 2;
    const base = boundaryCount;
    boundaryCount += perimeter;
    maximumCellCount = Math.max(maximumCellCount, count);
    rectangles.push(tile.column + column, tile.row + row, width, height, base);
    return {
      column: tile.column + column,
      row: tile.row + row,
      width,
      height,
      boundaryBase: base,
      boundaryCount: perimeter
    };
  });
  validateCapacity(
    id,
    'tile owned cell count',
    maximumCellCount,
    1,
    TERRAIN_TILED_MAXIMUM_FOREST_CELL_COUNT
  );
  validateCapacity(
    id,
    'coarse boundary cell count',
    boundaryCount,
    1,
    TERRAIN_TILED_MAXIMUM_FOREST_CELL_COUNT
  );
  // Sweep tile-edge row bands. Each band's column starts form a sorted partition, queried with
  // the shared binary search on the GPU; no invocation scans the list of tiles.
  const rowStarts = [...new Set(tiles.flatMap(tile => [tile.row, tile.row + tile.height]))].sort(
    (first, second) => first - second
  );
  const rowBands: number[] = [];
  const columnBands: number[] = [];
  for (const row of rowStarts.slice(0, -1)) {
    const owners = tiles
      .map((tile, index) => ({...tile, index}))
      .filter(tile => tile.row <= row && row < tile.row + tile.height)
      .sort((first, second) => first.column - second.column);
    rowBands.push(columnBands.length / 2, row);
    for (const tile of owners) columnBands.push(tile.index, tile.column);
    validateMaximumCount(
      id,
      'tile index words',
      columnBands.length,
      TERRAIN_TILED_MAXIMUM_INDEX_WORD_COUNT
    );
  }
  return {
    tiles,
    boundaryCount,
    maximumCellCount,
    rectangles: Uint32Array.from(rectangles),
    rowBands: Uint32Array.from(rowBands),
    columnBands: Uint32Array.from(columnBands),
    estimatedCommandNodeCount: getTerrainD8TileCommandNodeCount(props.tiles.length)
  };
}
export type TerrainD8TilePlan = ReturnType<typeof planTerrainD8Tiles>;

/** Every nested scratch plane and upload is checked before declaring the first resource. @internal */
export function validateTerrainD8TileBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  plan: TerrainD8TilePlan
): void {
  for (const [name, bytes] of [
    ['tile tour sums scratch', plan.maximumCellCount * 32],
    ['boundary tour sums scratch', plan.boundaryCount * 32],
    ['tile index rectangles', plan.rectangles.byteLength],
    ['tile index row bands', plan.rowBands.byteLength],
    ['tile index column bands', plan.columnBands.byteLength]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      bytes,
      'use fewer, larger tiles or reduce the largest tile dimensions'
    );
    validateCapacity(id, `${name} buffer bytes`, bytes, 1, graph.device.limits.maxBufferSize);
  }
}

/** Fixed geometry/index functions; all tile-specific values are immutable dispatch parameters. @internal */
export const TERRAIN_TILE_GEOMETRY_WGSL = `
const TILE_NONE: u32 = 0xffffffffu;
fn tileParameter(word: u32) -> u32 { return getChunkParameter(36u + word); }
fn tileWidth() -> u32 { return tileParameter(4u); }
fn tileHeight() -> u32 { return tileParameter(5u); }
fn localInput(cell: u32) -> u32 { return (cell / tileWidth() + tileParameter(3u)) * tileParameter(0u) + cell % tileWidth() + tileParameter(2u); }
fn globalCell(cell: u32) -> u32 { return (cell / tileWidth() + tileParameter(7u)) * GRID_WIDTH + cell % tileWidth() + tileParameter(6u); }
fn localCell(cell: u32) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 if (column < tileParameter(6u) || row < tileParameter(7u) || column - tileParameter(6u) >= tileWidth() || row - tileParameter(7u) >= tileHeight()) { return TILE_NONE; }
 return (row - tileParameter(7u)) * tileWidth() + column - tileParameter(6u);
}
fn isBoundary(cell: u32) -> bool { let column = cell % tileWidth(); let row = cell / tileWidth(); return column == 0u || row == 0u || column + 1u == tileWidth() || row + 1u == tileHeight(); }
fn boundaryLocal(slot: u32, width: u32, height: u32) -> u32 {
 if (width == 1u || height == 1u) { return slot; }
 if (slot < width) { return slot; }
 let middle = (height - 2u) * 2u;
 if (slot < width + middle) { let edge = slot - width; return (edge / 2u + 1u) * width + (edge % 2u) * (width - 1u); }
 return (height - 1u) * width + slot - width - middle;
}
fn boundarySlot(cell: u32, width: u32, height: u32) -> u32 {
 if (width == 1u || height == 1u) { return cell; }
 let column = cell % width; let row = cell / width;
 if (row == 0u) { return column; }
 if (row + 1u == height) { return width + (height - 2u) * 2u + column; }
 return width + (row - 1u) * 2u + u32(column != 0u);
}
fn localNeighbor(cell: u32, direction: u32) -> u32 {
 let global = getFlowReceiver(globalCell(cell), 1u << direction);
 if (global == NO_CELL) { return TILE_NONE; }
 return localCell(global);
}
fn isStream(word: u32) -> bool { return word != 0u && word < UNRESOLVED_DIRECTION; }
`;

/** rowBands/columnBands use (low payload, high key), the shared unsigned-64 search layout. @internal */
export const TERRAIN_TILE_LOOKUP_WGSL = `${getBinarySearchWGSL('rowBands', 'u64')}
${getBinarySearchWGSL('columnBands', 'u64')}
fn globalBoundary(cell: u32) -> u32 {
 let column = cell % GRID_WIDTH; let row = cell / GRID_WIDTH;
 let bandCount = (arrayLength(&rowBands) - rowBandsOffset) / 2u;
 let band = upperBoundRowBands(0u, bandCount, vec2<u32>(row, TILE_NONE)) - 1u;
 let first = rowBands[rowBandsOffset + band * 2u];
 var last = (arrayLength(&columnBands) - columnBandsOffset) / 2u;
 if (band + 1u < bandCount) { last = rowBands[rowBandsOffset + (band + 1u) * 2u]; }
 let interval = upperBoundColumnBands(first, last, vec2<u32>(column, TILE_NONE)) - 1u;
 let tile = columnBands[columnBandsOffset + interval * 2u] * 5u + rectanglesOffset;
 let width = rectangles[tile + 2u]; let height = rectangles[tile + 3u];
 let local = (row - rectangles[tile + 1u]) * width + column - rectangles[tile];
 return rectangles[tile + 4u] + boundarySlot(local, width, height);
}
`;
