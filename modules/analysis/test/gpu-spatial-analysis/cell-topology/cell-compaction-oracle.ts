// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU reference for GPUCellCompaction. Quadbin follows CARTO quadbin with BigInt; H3 uses h3-js.

import {cellToChildren, isPentagon} from 'h3-js';
import {
  bigIntToH3,
  getCellParent as getExistingCellParent,
  getCellResolution as getExistingCellResolution,
  h3ToBigInt,
  isValidCellKey as isExistingValidCellKey,
  quadbinCellToTile,
  quadbinTileToCell,
  type CellFamily as ExistingCellFamily
} from '../cell-aggregation/cell-aggregation-oracle';

import {getS2TopologyRowOnCPU} from './s2-topology-oracle';

export type CellCompactionFamily = ExistingCellFamily | 's2';

function isValidCellKey(family: CellCompactionFamily, cell: bigint): boolean {
  return family === 's2'
    ? cell > 0n &&
        cell < 1n << 64n &&
        cell >> 61n < 6n &&
        (cell & -cell & 0x1555555555555555n) !== 0n
    : isExistingValidCellKey(family, cell);
}

function getCellResolution(family: CellCompactionFamily, cell: bigint): number {
  if (family !== 's2') return getExistingCellResolution(family, cell);
  let marker = cell & -cell;
  let level = 30;
  while (marker > 1n) {
    marker /= 4n;
    level--;
  }
  return level;
}

function getCellParent(family: CellCompactionFamily, cell: bigint, resolution: number): bigint {
  return family === 's2'
    ? getS2TopologyRowOnCPU({type: 'parent', resolution}, cell).cells[0]
    : getExistingCellParent(family, cell, resolution);
}

/** Ascending BigInt comparator. */
export function compareBigInt(left: bigint, right: bigint): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/** Descendants of `cell` at `resolution`, in ascending key order. */
export function getCellDescendants(
  family: CellCompactionFamily,
  cell: bigint,
  resolution: number
): bigint[] {
  let result: bigint[];
  if (family === 'quadbin') {
    const {x, y, z} = quadbinCellToTile(cell);
    const side = 2 ** (resolution - z);
    result = [];
    for (let dx = 0; dx < side; dx++) {
      for (let dy = 0; dy < side; dy++) {
        result.push(quadbinTileToCell(x * side + dx, y * side + dy, resolution));
      }
    }
  } else if (family === 's2') {
    result = getS2TopologyRowOnCPU(
      {type: 'children', inputResolution: getCellResolution(family, cell), resolution},
      cell
    ).cells;
  } else {
    result = cellToChildren(bigIntToH3(cell), resolution).map(h3ToBigInt);
  }
  return result.sort(compareBigInt);
}

/** Number of children of a cell: 4 (Quadbin), 7 (H3 hexagon) or 6 (H3 pentagon). */
export function getChildCount(family: CellCompactionFamily, cell: bigint): number {
  if (family !== 'h3') {
    return 4;
  }
  return isPentagon(bigIntToH3(cell)) ? 6 : 7;
}

/**
 * Minimal cover: valid cells at `resolution` are deduplicated, then every complete child set is
 * replaced by its parent down to `minimumResolution`. Returns ascending keys.
 */
export function compactCellsOnCPU(
  family: CellCompactionFamily,
  cells: readonly bigint[],
  resolution: number,
  minimumResolution = 0
): bigint[] {
  const result: bigint[] = [];
  let level = [
    ...new Set(
      cells.filter(
        cell => isValidCellKey(family, cell) && getCellResolution(family, cell) === resolution
      )
    )
  ];
  for (let current = resolution; current > minimumResolution; current--) {
    const groups = new Map<bigint, bigint[]>();
    for (const cell of level) {
      const parent = getCellParent(family, cell, current - 1);
      groups.set(parent, [...(groups.get(parent) ?? []), cell]);
    }
    const next: bigint[] = [];
    for (const [parent, children] of groups) {
      if (children.length === getChildCount(family, parent)) {
        next.push(parent);
      } else {
        result.push(...children);
      }
    }
    level = next;
  }
  result.push(...level);
  return result.sort(compareBigInt);
}

/** Expands each valid cell at or above `resolution` (within `maximumDepth`), in input order. */
export function uncompactCellsOnCPU(
  family: CellCompactionFamily,
  cells: readonly bigint[],
  resolution: number,
  maximumDepth = 8
): bigint[] {
  const result: bigint[] = [];
  for (const cell of cells) {
    if (!isValidCellKey(family, cell)) {
      continue;
    }
    const cellResolution = getCellResolution(family, cell);
    if (cellResolution <= resolution && resolution - cellResolution <= maximumDepth) {
      result.push(...getCellDescendants(family, cell, resolution));
    }
  }
  return result;
}
