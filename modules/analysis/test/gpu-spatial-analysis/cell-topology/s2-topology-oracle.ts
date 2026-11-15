// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCellTopologyOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import reference from '../global-coordinates/fixtures/global-coordinates.json';
import type {CellTopologyRow} from './cell-topology-oracle';
import {s2CellFromFaceIJ, s2CellToFaceIJ} from '../cell-indexing/cell-indexing-oracle';

/** Independent s2sphere 0.2.5 graph; complete at levels 0, 1 and 2. */
export const S2_TOPOLOGY_REFERENCE = reference.cells.map(row => ({
  cell: BigInt(row.cell[0]) | (BigInt(row.cell[1]) << 32n),
  level: row.level,
  neighbors: row.neighbors.map(words => BigInt(words[0]) | (BigInt(words[1]) << 32n))
}));
const NEIGHBORS = new Map(S2_TOPOLOGY_REFERENCE.map(row => [row.cell, row.neighbors]));

/** Browser-safe key construction, cross-checked against math.gl in the node spec. */
export function getS2TopologyCell({
  face,
  ij: [column, row],
  level
}: {
  face: number;
  ij: readonly [number, number];
  level: number;
}): bigint {
  const leafScale = 2 ** (30 - level);
  return s2CellFromFaceIJ(face, column * leafScale, row * leafScale, level);
}

function isValidS2ReferenceCell(cell: bigint): boolean {
  return (
    cell > 0n && cell < 1n << 64n && cell >> 61n < 6n && (cell & -cell & 0x1555555555555555n) !== 0n
  );
}

/** The four child centers are symmetric about their parent in Hilbert order. */
function getS2ReferenceChildren(cell: bigint): bigint[] {
  const childMarker = (cell & -cell) / 4n;
  return [-3n, -1n, 1n, 3n].map(offset => cell + offset * childMarker);
}

/**
 * Browser-safe exact-key oracle: pinned s2sphere face transitions, interior grid steps and
 * BigInt hierarchy traversal. The node spec verifies construction and hierarchy against
 * published math.gl through level 30, without loading its aggregate entry in the browser.
 */
export function getS2TopologyRowOnCPU(
  operation: GPUCellTopologyOperation,
  cell: bigint
): CellTopologyRow {
  if (!isValidS2ReferenceCell(cell)) return {cells: [], distances: []};
  const level = s2CellToFaceIJ(cell).level;
  if (operation.type === 'parent') {
    if (level < operation.resolution) return {cells: [], distances: []};
    // Traverse the independently encoded Hilbert hierarchy from the face root, selecting the
    // child whose leaf interval contains the input. This does not reproduce the GPU bit mask.
    let parent = ((cell >> 61n) << 61n) | (1n << 60n);
    for (let depth = 1; depth <= operation.resolution; depth++) {
      const children = getS2ReferenceChildren(parent);
      parent = children.find(child => {
        const marker = child & -child;
        return child - marker < cell && cell < child + marker;
      })!;
    }
    return {cells: [parent], distances: [0]};
  }
  if (operation.type === 'children') {
    if (level !== operation.inputResolution) return {cells: [], distances: []};
    let children = [cell];
    for (let depth = level; depth < operation.resolution; depth++)
      children = children.flatMap(getS2ReferenceChildren);
    return {cells: children, distances: children.map(() => 0)};
  }
  const distances = new Map([[cell, 0]]);
  let frontier = [cell];
  for (let distance = 1; distance <= operation.k; distance++) {
    const next: bigint[] = [];
    for (const current of frontier) {
      let neighbors = NEIGHBORS.get(current);
      if (!neighbors) {
        const {face, i: column, j: row, level} = s2CellToFaceIJ(current);
        const maximum = 2 ** level - 1;
        if (column <= 0 || row <= 0 || column >= maximum || row >= maximum)
          throw new Error(`s2sphere fixture lacks face-edge neighbors for ${current.toString(16)}`);
        neighbors = [
          [column - 1, row],
          [column + 1, row],
          [column, row - 1],
          [column, row + 1]
        ].map(([column, row]) => getS2TopologyCell({face, ij: [column, row], level}));
      }
      for (const neighbor of neighbors) {
        if (!distances.has(neighbor)) {
          distances.set(neighbor, distance);
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  const entries = [...distances].filter(([, distance]) =>
    operation.type === 'ring' ? distance === operation.k : true
  );
  entries.sort(
    ([firstKey, firstDistance], [secondKey, secondDistance]) =>
      firstDistance - secondDistance || (firstKey < secondKey ? -1 : firstKey > secondKey ? 1 : 0)
  );
  return {cells: entries.map(([key]) => key), distances: entries.map(([, distance]) => distance)};
}
