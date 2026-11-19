// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Six hydrography features (a lake polygon, its outflow river and a gauge point), delivered twice
// in the postorder tree layout of GPUNormalizeGeometry. Delivery A stores every feature the same
// way. Delivery B stores the first three differently without changing them (part order, ring
// start and winding, line direction) and edits the last three (a vertex moved, a vertex added, a
// part missing). The shapes are the sampled lake profiles and river reaches of ../../../../data.

import {GPU_CANONICAL_GEOMETRY_TYPE} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createRiverLine, createShoreRing} from '../../../../data/hydrography';

const {point, lineString, linearRing, polygon, collection} = GPU_CANONICAL_GEOMETRY_TYPE;

const COLUMN_COUNT = 3;
const ROW_COUNT = 2;
/**
 * The features are laid out in the middle of the picture, at most this wide for a world of height
 * one: a 16:9 poster is cropped to a 16:10 card, and nothing may be lost at its sides.
 */
const MAXIMUM_LAYOUT_ASPECT = 1.6;
/** Sampling phases tried for every lake ring. */
const PHASE_COUNT = 16;
const RIVER_VERTEX_COUNT = 7;
/**
 * The lake of each feature: a shore profile of ../../../../data sampled to `vertexCount` unique
 * vertices (the stored ring repeats its first vertex to close) at `radius` world units.
 */
const LAKES = [
  {profile: 0, vertexCount: 10, radius: 0.195},
  {profile: 3, vertexCount: 10, radius: 0.132},
  {profile: 1, vertexCount: 10, radius: 0.15},
  {profile: 7, vertexCount: 10, radius: 0.144},
  {profile: 6, vertexCount: 10, radius: 0.172},
  {profile: 2, vertexCount: 7, radius: 0.172}
];
/** River reach of each feature (the two-vertex Snake reach is left out). */
const RIVER_REACHES = [0, 1, 2, 4, 5, 1];
/** Row centers as fractions of the world height, top row first. */
const ROW_CENTERS = [0.655, 0.235];
/** The label of each feature sits this far above its row center. */
const LABEL_RISE = 0.185;

/** How delivery B differs from delivery A, one variant per feature. */
export const DELIVERY_VARIANTS = [
  {id: 'reordered', label: 'parts reordered', isEdit: false},
  {id: 'restarted', label: 'ring restarted', isEdit: false},
  {id: 'reversed', label: 'line reversed', isEdit: false},
  {id: 'moved', label: 'vertex moved', isEdit: true},
  {id: 'added', label: 'vertex added', isEdit: true},
  {id: 'missing', label: 'part missing', isEdit: true}
] as const;

export const FEATURE_COUNT = DELIVERY_VARIANTS.length;

type World = {width: number; height: number};
export type PartName = 'lake' | 'river' | 'gauge';
const PART_NAMES: readonly PartName[] = ['lake', 'river', 'gauge'];

/** A stored line, ring or point: the rows `[firstRow, firstRow + rowCount)` of a delivery. */
export type DeliveryLeaf = {feature: number; part: PartName; firstRow: number; rowCount: number};

/** One delivery as the operation receives it; positions are written by `writeDeliveryPositions`. */
export type DeliveryTree = {
  rowCount: number;
  /** One `[type, firstVertexOrChildSlot, count, 0]` row per node, children before parents. */
  nodes: Uint32Array;
  children: Uint32Array;
  /** The collection node of every feature: what a pair names. */
  roots: Uint32Array;
  /** Consecutive vertex rows of every line and ring, for drawing. */
  edges: Uint32Array;
  featureOfRow: Uint32Array;
  /** The vertex (an index into `homes`) every stored row holds. */
  vertexOfRow: Uint32Array;
  leaves: DeliveryLeaf[];
  /** Stored part names of every feature, in stored order. */
  partOrder: PartName[][];
};

export type FeatureDeliveries = {
  /** Where every vertex is in delivery A, as (x, y). */
  homes: Float32Array;
  featureOfVertex: Uint32Array;
  /** The number of vertices delivery A has (delivery B adds one after them). */
  sharedCount: number;
  /** The vertices both deliveries store: the ones a reader can carry. */
  carriable: number[];
  /** The vertex that delivery B stores displaced, in the "vertex moved" feature. */
  movedVertex: number;
  /** The vertex only delivery B has, and the two vertices it lies between. */
  addedVertex: {vertex: number; between: readonly [number, number]};
  left: DeliveryTree;
  right: DeliveryTree;
  /** Where each feature's label goes, as (x, y). */
  labelAnchors: Float32Array;
  /** The top left corner of the layout: where the key starts. */
  keyAnchor: readonly [number, number];
  /** Where a note about one part of a feature goes: `partAnchors[feature][part]`. */
  partAnchors: Record<PartName, readonly [number, number]>[];
  /** The smallest gap between a ring's first and second smallest x: see `getStartGap`. */
  startGap: number;
};

/**
 * Samples a shore profile so that one vertex is clearly the westernmost. GPUNormalizeGeometry
 * starts a ring at its lexicographically smallest vertex, so a ring whose two westernmost vertices
 * are closer in x than the tolerance could start elsewhere after an edit smaller than the tolerance.
 * The ring is returned with its bounding box's west edge at x = 0 and its middle at y = 0.
 */
function createStableShoreRing(lake: (typeof LAKES)[number]): number[][] {
  let best: number[][] = [];
  let bestGap = -Infinity;
  for (let step = 0; step < PHASE_COUNT; step++) {
    // createShoreRing starts at angle zero: sampling more densely and keeping every
    // PHASE_COUNT-th vertex from `step` tries that many phases of a ring of this vertex count.
    const ring = createShoreRing(
      lake.profile,
      0,
      0,
      lake.radius,
      lake.vertexCount * PHASE_COUNT
    ).filter((_, vertex) => vertex % PHASE_COUNT === step);
    const gap = getStartGap(ring);
    if (gap > bestGap) {
      bestGap = gap;
      best = ring;
    }
  }
  const eastings = best.map(([x]) => x);
  const northings = best.map(([, y]) => y);
  const west = Math.min(...eastings);
  const north = Math.max(...northings);
  const middle = (Math.min(...northings) + north) / 2;
  // Delivery A stores the ring from its northernmost vertex, away from where the river leaves.
  const start = northings.indexOf(north);
  return best.map((_, vertex) => {
    const [x, y] = best[(start + vertex) % best.length];
    return [x - west, y - middle];
  });
}

/** The gap in x between the westernmost vertex of a ring and the next one. */
function getStartGap(ring: number[][]): number {
  const eastings = ring.map(([x]) => x).sort((first, second) => first - second);
  return eastings[1] - eastings[0];
}

type FeatureParts = {lake: number[]; river: number[]; gauge: number[]};

type StoredFeature = {
  /** Ring vertices as stored, without the closing repeat. */
  lake: number[];
  river: number[];
  gauge: number[];
  order: PartName[];
};

function buildTree(features: StoredFeature[]): DeliveryTree {
  const nodes: number[] = [];
  const children: number[] = [];
  const roots: number[] = [];
  const edges: number[] = [];
  const featureOfRow: number[] = [];
  const vertexOfRow: number[] = [];
  const leaves: DeliveryLeaf[] = [];
  const addLeaf = (type: number, vertices: number[], feature: number, part: PartName): number => {
    const first = vertexOfRow.length;
    for (const vertex of vertices) {
      vertexOfRow.push(vertex);
      featureOfRow.push(feature);
    }
    for (let row = first; row + 1 < first + vertices.length; row++) edges.push(row, row + 1);
    leaves.push({feature, part, firstRow: first, rowCount: vertices.length});
    nodes.push(type, first, vertices.length, 0);
    return nodes.length / 4 - 1;
  };
  const addParent = (type: number, childNodes: number[]): number => {
    nodes.push(type, children.length, childNodes.length, 0);
    children.push(...childNodes);
    return nodes.length / 4 - 1;
  };
  features.forEach((feature, featureIndex) => {
    const parts = feature.order.map(part => {
      if (part === 'lake') {
        const ring = addLeaf(linearRing, [...feature.lake, feature.lake[0]], featureIndex, part);
        return addParent(polygon, [ring]);
      }
      return part === 'river'
        ? addLeaf(lineString, feature.river, featureIndex, part)
        : addLeaf(point, feature.gauge, featureIndex, part);
    });
    roots.push(addParent(collection, parts));
  });
  return {
    rowCount: vertexOfRow.length,
    nodes: Uint32Array.from(nodes),
    children: Uint32Array.from(children),
    roots: Uint32Array.from(roots),
    edges: Uint32Array.from(edges),
    featureOfRow: Uint32Array.from(featureOfRow),
    vertexOfRow: Uint32Array.from(vertexOfRow),
    leaves,
    partOrder: features.map(feature => feature.order)
  };
}

/** Builds both deliveries. The same world always gives the same features. */
export function createFeatureDeliveries(world: World): FeatureDeliveries {
  const homes: number[] = [];
  const featureOfVertex: number[] = [];
  const labelAnchors: number[] = [];
  const partAnchors: Record<PartName, readonly [number, number]>[] = [];
  const parts: FeatureParts[] = [];
  let startGap = Infinity;
  const layoutWidth = Math.min(world.width, MAXIMUM_LAYOUT_ASPECT * world.height);
  const layoutLeft = (world.width - layoutWidth) / 2;
  const cellWidth = layoutWidth / COLUMN_COUNT;
  for (let feature = 0; feature < FEATURE_COUNT; feature++) {
    const cellLeft = layoutLeft + (feature % COLUMN_COUNT) * cellWidth;
    const centerY = ROW_CENTERS[Math.floor(feature / COLUMN_COUNT) % ROW_COUNT] * world.height;
    const ring = createStableShoreRing(LAKES[feature]).map(([x, y]) => [
      cellLeft + 0.035 + x,
      centerY + y
    ]);
    const addVertices = (vertices: number[][]): number[] =>
      vertices.map(([x, y]) => {
        homes.push(x, y);
        featureOfVertex.push(feature);
        return featureOfVertex.length - 1;
      });
    startGap = Math.min(startGap, getStartGap(ring));
    // The river leaves the lake at its easternmost vertex and runs to the edge of the cell.
    const outlet = ring.reduce((best, vertex) => (vertex[0] > best[0] ? vertex : best));
    const riverLength = cellLeft + cellWidth - 0.04 - (outlet[0] + 0.02);
    const reach = createRiverLine(RIVER_REACHES[feature], RIVER_VERTEX_COUNT, {
      width: riverLength / 0.93,
      height: 0.15 / 0.85
    });
    const river = reach.map(([x, y]) => [
      outlet[0] + 0.02 + x - reach[0][0],
      outlet[1] + y - reach[0][1]
    ]);
    const mouth = river[river.length - 1];
    const gauge = [[mouth[0] - 0.035, Math.min(mouth[1], centerY) - 0.085]];
    parts.push({lake: addVertices(ring), river: addVertices(river), gauge: addVertices(gauge)});
    labelAnchors.push(cellLeft + cellWidth / 2, centerY + LABEL_RISE);
    const middle = river[Math.floor(river.length / 2)];
    partAnchors.push({
      lake: [
        (outlet[0] + cellLeft + 0.035) / 2,
        ring.reduce((sum, vertex) => sum + vertex[1], 0) / ring.length
      ],
      river: [middle[0], middle[1] + 0.055],
      gauge: [gauge[0][0] - 0.03, gauge[0][1]]
    });
  }
  const sharedCount = featureOfVertex.length;
  const variantIndex = (id: (typeof DELIVERY_VARIANTS)[number]['id']) =>
    DELIVERY_VARIANTS.findIndex(variant => variant.id === id);

  // The moved vertex is the northernmost of its lake: it is displaced east, so the displacement
  // is all in one coordinate and compares directly with the tolerance.
  const movedLake = parts[variantIndex('moved')].lake;
  const movedVertex = movedLake.reduce((best, vertex) =>
    homes[vertex * 2 + 1] > homes[best * 2 + 1] ? vertex : best
  );
  // The added vertex is the midpoint of a river edge: the line keeps its shape.
  const addedRiver = parts[variantIndex('added')].river;
  const between = [addedRiver[3], addedRiver[4]] as const;
  homes.push(
    (homes[between[0] * 2] + homes[between[1] * 2]) / 2,
    (homes[between[0] * 2 + 1] + homes[between[1] * 2 + 1]) / 2
  );
  featureOfVertex.push(variantIndex('added'));
  const addedVertex = {vertex: sharedCount, between};

  const order = [...PART_NAMES];
  const left = buildTree(parts.map(feature => ({...feature, order})));
  const right = buildTree(
    parts.map((feature, featureIndex): StoredFeature => {
      switch (DELIVERY_VARIANTS[featureIndex].id) {
        case 'reordered':
          return {...feature, order: ['river', 'gauge', 'lake']};
        case 'restarted':
          // Starts three vertices later and winds the other way.
          return {
            ...feature,
            lake: feature.lake.map(
              (_, vertex) => feature.lake[(3 + feature.lake.length - vertex) % feature.lake.length]
            ),
            order
          };
        case 'reversed':
          return {...feature, river: [...feature.river].reverse(), order};
        case 'added':
          return {
            ...feature,
            river: [...feature.river.slice(0, 4), addedVertex.vertex, ...feature.river.slice(4)],
            order
          };
        case 'missing':
          return {...feature, order: ['lake', 'river']};
        default:
          return {...feature, order};
      }
    })
  );
  return {
    homes: Float32Array.from(homes),
    featureOfVertex: Uint32Array.from(featureOfVertex),
    sharedCount,
    carriable: Array.from({length: sharedCount}, (_, vertex) => vertex).filter(vertex =>
      right.vertexOfRow.includes(vertex)
    ),
    movedVertex,
    addedVertex,
    left,
    right,
    labelAnchors: Float32Array.from(labelAnchors),
    keyAnchor: [layoutLeft + 0.035, world.height],
    partAnchors,
    startGap
  };
}

/** A vertex that is somewhere else than its home: the carried vertex, or the moved one. */
export type VertexOverride = {vertex: number; at: readonly [number, number]};

/** Where a vertex of delivery B is, given the vertices that are not at home. */
export function getDeliveryVertex(
  deliveries: FeatureDeliveries,
  vertex: number,
  overrides: readonly VertexOverride[]
): [number, number] {
  const {addedVertex, homes} = deliveries;
  const override = overrides.find(candidate => candidate.vertex === vertex);
  if (override) return [override.at[0], override.at[1]];
  if (vertex === addedVertex.vertex) {
    // The added vertex stays the midpoint of its edge when an end of that edge is carried.
    const first = getDeliveryVertex(deliveries, addedVertex.between[0], overrides);
    const second = getDeliveryVertex(deliveries, addedVertex.between[1], overrides);
    return [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2];
  }
  return [homes[vertex * 2], homes[vertex * 2 + 1]];
}

/** The stored positions of a delivery: every row at its vertex's home unless overridden. */
export function writeDeliveryPositions(
  deliveries: FeatureDeliveries,
  tree: DeliveryTree,
  overrides: readonly VertexOverride[] = []
): Float32Array {
  const positions = new Float32Array(tree.rowCount * 2);
  for (let row = 0; row < tree.rowCount; row++) {
    const [x, y] = getDeliveryVertex(deliveries, tree.vertexOfRow[row], overrides);
    positions[row * 2] = x;
    positions[row * 2 + 1] = y;
  }
  return positions;
}
