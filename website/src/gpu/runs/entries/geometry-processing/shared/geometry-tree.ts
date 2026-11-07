// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A small geometry collection in the postorder tree layout of GPUNormalizeGeometry: leaves (rings,
// lines, points) first, then one collection root whose children are listed in a shuffled order.

import {makeRandom} from '../../../../demos/kit';
import type {World} from './polygon-set';

const POINT = 0;
const LINE_STRING = 2;
const LINEAR_RING = 3;
const COLLECTION = 7;

const LEAF_KINDS = [LINEAR_RING, LINE_STRING, POINT] as const;
const LEAF_COLUMNS = 4;
const LEAF_ROWS = 3;

export type GeometryTreeOptions = {
  /** Seed of the order of the root's children. */
  shuffleSeed: number;
  /** Where every ring starts (rings are rotated by this many vertices). */
  start: number;
  /** Leaf `k` is displaced by `k * displacement` along x (0 leaves the tree as is). */
  displacement?: number;
};

export type GeometryTree = {
  leafCount: number;
  positions: Float32Array;
  /** One `[type, first, count, 0]` row per node; the root is the last row. */
  nodes: Uint32Array;
  children: Uint32Array;
  rootId: number;
  /** Consecutive vertex id pairs of every line and ring leaf (closing edge for rings). */
  edges: Uint32Array;
  /** The leaf row of every vertex row. */
  leafOfVertex: Uint32Array;
};

/** Builds the collection; the same shapes come back for every option, only order and starts change. */
export function createGeometryTree(world: World, options: GeometryTreeOptions): GeometryTree {
  const random = makeRandom(7);
  const leafCount = LEAF_COLUMNS * LEAF_ROWS;
  const positions: number[] = [];
  const nodes: number[] = [];
  const edges: number[] = [];
  const leafOfVertex: number[] = [];
  for (let leaf = 0; leaf < leafCount; leaf++) {
    const kind = LEAF_KINDS[leaf % LEAF_KINDS.length];
    const column = leaf % LEAF_COLUMNS;
    const row = Math.floor(leaf / LEAF_COLUMNS);
    const centerX = ((column + 0.5) / LEAF_COLUMNS) * world.width;
    const centerY = ((row + 0.5) / LEAF_ROWS) * world.height;
    const radius = 0.11 * (0.8 + 0.4 * random());
    const first = positions.length / 2;
    const vertices: number[][] = [];
    if (kind === LINEAR_RING) {
      const count = 7;
      for (let vertex = 0; vertex < count; vertex++) {
        const angle = (vertex / count) * Math.PI * 2;
        const spike = vertex % 2 === 0 ? 1 : 0.6;
        vertices.push([
          centerX + Math.cos(angle) * radius * spike,
          centerY + Math.sin(angle) * radius * spike
        ]);
      }
      const shift = (leaf + options.start) % count;
      const rotated = vertices.map((_, index) => vertices[(index + shift) % count]);
      vertices.splice(0, vertices.length, ...rotated, rotated[0]);
    } else if (kind === LINE_STRING) {
      for (let vertex = 0; vertex < 6; vertex++) {
        vertices.push([
          centerX + (vertex / 5 - 0.5) * radius * 2.4,
          centerY + (vertex % 2 === 0 ? -1 : 1) * radius * 0.5
        ]);
      }
    } else {
      vertices.push([centerX, centerY]);
    }
    const shift = (options.displacement ?? 0) * leaf;
    for (const [x, y] of vertices) {
      positions.push(x + shift, y);
      leafOfVertex.push(leaf);
    }
    nodes.push(kind, first, vertices.length, 0);
    if (kind !== POINT) {
      for (let vertex = 0; vertex + 1 < vertices.length; vertex++)
        edges.push(first + vertex, first + vertex + 1);
    }
  }
  // The root collection lists its children in a shuffled order.
  const shuffled = Array.from({length: leafCount}, (_, index) => index);
  const shuffler = makeRandom(options.shuffleSeed + 11);
  for (let index = shuffled.length - 1; index > 0; index--) {
    const swap = Math.floor(shuffler() * (index + 1));
    [shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]];
  }
  nodes.push(COLLECTION, 0, leafCount, 0);
  return {
    leafCount,
    positions: Float32Array.from(positions),
    nodes: Uint32Array.from(nodes),
    children: Uint32Array.from(shuffled),
    rootId: leafCount,
    edges: Uint32Array.from(edges),
    leafOfVertex: Uint32Array.from(leafOfVertex)
  };
}
