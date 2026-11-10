// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTracts, type CsrWeights, type Random, type World} from '../../../inputs';
import {PolygonBuilder} from './polygons';

/** A connected window of Census tract shells, counterclockwise for the coverage operations. */
export function createTractCoverage(random: Random, world: World, rows: number) {
  const tracts = createTracts(random, world, rows);
  const builder = new PolygonBuilder();
  for (let feature = 0; feature < tracts.count; feature++) {
    const ring = [];
    for (
      let vertex = tracts.ringOffsets[feature + 1] - 1;
      vertex >= tracts.ringOffsets[feature];
      vertex--
    ) {
      ring.push([tracts.vertices[vertex * 2], tracts.vertices[vertex * 2 + 1]]);
    }
    builder.addFeature([ring]);
  }
  return {...builder.build(), centers: tracts.centers, contiguity: tracts.contiguity};
}

/** Model contiguous districts by growing each label along actual shared tract edges. */
export function groupNeighborTracts(contiguity: CsrWeights, size: number): Uint32Array {
  const labels = new Uint32Array(contiguity.rowCount).fill(0xffffffff);
  let group = 0;
  for (let start = 0; start < labels.length; start++) {
    if (labels[start] !== 0xffffffff) continue;
    const queue = [start];
    labels[start] = group;
    for (let next = 0; next < queue.length && queue.length < size; next++) {
      const row = queue[next];
      for (
        let slot = contiguity.offsets[row];
        slot < contiguity.offsets[row + 1] && queue.length < size;
        slot++
      ) {
        const neighbor = contiguity.neighbors[slot];
        if (labels[neighbor] === 0xffffffff) {
          labels[neighbor] = group;
          queue.push(neighbor);
        }
      }
    }
    group++;
  }
  return labels;
}

function getEdgeHash(index: number): number {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

/**
 * A coverage of grid cells whose shared boundaries wiggle: each edge between two lattice points is
 * one polyline used by both neighbors, so the coverage is valid and gap-free, and simplifying it
 * visibly straightens the boundaries.
 */
export function createWigglyCoverage(world: World, columns: number, rows: number, wiggle = 0.03) {
  const margin = 0.07;
  const cellWidth = (world.width - 2 * margin) / columns;
  const cellHeight = (1 - 2 * margin) / rows;
  const lattice = (column: number, row: number): [number, number] => {
    const isInterior = column > 0 && column < columns && row > 0 && row < rows;
    const index = row * (columns + 1) + column;
    return [
      margin + column * cellWidth + (isInterior ? (getEdgeHash(index) - 0.5) * cellWidth * 0.3 : 0),
      margin +
        row * cellHeight +
        (isInterior ? (getEdgeHash(index + 500) - 0.5) * cellHeight * 0.3 : 0)
    ];
  };
  const vertexCount = 8;
  // Points of the edge from `from` to `to` without its last point, pushed sideways by noise.
  const createEdge = (from: [number, number], to: [number, number], seed: number) => {
    const deltaX = to[0] - from[0];
    const deltaY = to[1] - from[1];
    const length = Math.hypot(deltaX, deltaY);
    return Array.from({length: vertexCount}, (_, step) => {
      const fraction = step / vertexCount;
      const push = step === 0 ? 0 : (getEdgeHash(seed * 31 + step) - 0.5) * 2 * wiggle;
      return [
        from[0] + deltaX * fraction - (deltaY / length) * push,
        from[1] + deltaY * fraction + (deltaX / length) * push
      ];
    });
  };
  const horizontal = new Map<number, number[][]>();
  const vertical = new Map<number, number[][]>();
  const getHorizontal = (column: number, row: number) => {
    const key = row * columns + column;
    if (!horizontal.has(key)) {
      horizontal.set(key, createEdge(lattice(column, row), lattice(column + 1, row), key));
    }
    return horizontal.get(key)!;
  };
  const getVertical = (column: number, row: number) => {
    const key = row * (columns + 1) + column;
    if (!vertical.has(key)) {
      vertical.set(key, createEdge(lattice(column, row), lattice(column, row + 1), key + 4000));
    }
    return vertical.get(key)!;
  };
  const builder = new PolygonBuilder();
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      // Counterclockwise: bottom left to right, right bottom to top, top right to left, left top
      // to bottom. A reversed edge starts at its far lattice point and stops before its first.
      const reverseEdge = (edge: number[][], end: [number, number]) => [
        [end[0], end[1]],
        ...edge.slice(1).reverse()
      ];
      builder.addFeature([
        [
          ...getHorizontal(column, row),
          ...getVertical(column + 1, row),
          ...reverseEdge(getHorizontal(column, row + 1), lattice(column + 1, row + 1)),
          ...reverseEdge(getVertical(column, row), lattice(column, row + 1))
        ]
      ]);
    }
  }
  return builder.build();
}
