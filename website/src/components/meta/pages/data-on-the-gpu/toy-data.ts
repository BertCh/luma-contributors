// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Fixed toy data for the figures of "Data on the GPU". The flat arrays are built from the nested
// shapes, so the offsets on the page are computed and cannot disagree with the pictures.

export type ToyPoint = readonly [number, number];

/** A toy feature: a list of polygons, each a list of rings (the first ring is the shell). */
export type ToyFeature = {label: string; polygons: readonly (readonly (readonly ToyPoint[])[])[]};

export const TOY_FEATURES: readonly ToyFeature[] = [
  {
    label: 'simple',
    polygons: [
      [
        [
          [1, 1],
          [9, 1],
          [9, 9],
          [1, 9]
        ]
      ]
    ]
  },
  {
    label: 'with a hole',
    polygons: [
      [
        [
          [13, 1],
          [23, 1],
          [23, 9],
          [13, 9]
        ],
        [
          [16, 3],
          [16, 7],
          [20, 7],
          [20, 3]
        ]
      ]
    ]
  },
  {
    label: 'two parts',
    polygons: [
      [
        [
          [26, 1],
          [31, 1],
          [26, 6]
        ]
      ],
      [
        [
          [33, 4],
          [38, 4],
          [38, 9]
        ]
      ]
    ]
  }
];

export type ToyRing = {
  feature: number;
  polygon: number;
  ring: number;
  /** First vertex, inclusive. */
  start: number;
  /** One past the last vertex. */
  end: number;
};

export type ToyPolygonLayer = {
  positions: ToyPoint[];
  /** Feature-to-polygon offsets with one terminal entry (`featureOffsets`). */
  featureOffsets: number[];
  /** Polygon-to-ring offsets with one terminal entry (`polygonOffsets`). */
  polygonOffsets: number[];
  /** Ring-to-vertex offsets with one terminal entry (`ringOffsets`). */
  ringOffsets: number[];
  rings: ToyRing[];
};

/** Flattens nested features into the four arrays of `GPUPolygonGeometryPort`. */
export function buildPolygonLayer(features: readonly ToyFeature[]): ToyPolygonLayer {
  const positions: ToyPoint[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  const rings: ToyRing[] = [];
  let polygonCount = 0;
  let ringCount = 0;
  features.forEach((feature, featureIndex) => {
    for (const polygon of feature.polygons) {
      for (const ring of polygon) {
        const start = positions.length;
        positions.push(...ring);
        rings.push({
          feature: featureIndex,
          polygon: polygonCount,
          ring: ringCount,
          start,
          end: positions.length
        });
        ringOffsets.push(positions.length);
        ringCount++;
      }
      polygonOffsets.push(ringCount);
      polygonCount++;
    }
    featureOffsets.push(polygonCount);
  });
  return {positions, featureOffsets, polygonOffsets, ringOffsets, rings};
}

export const TOY_POLYGON_LAYER = buildPolygonLayer(TOY_FEATURES);

export type ToyGraph = {
  nodes: readonly ToyPoint[];
  edges: readonly (readonly [number, number])[];
  /** `nodeCount + 1` row offsets (`offsets`). */
  offsets: number[];
  /** Target node per edge slot (`neighbors`). */
  neighbors: number[];
  /** Row-standardised weight per edge slot, one over the row's neighbour count (`weights`). */
  weights: number[];
};

const TOY_NODES: readonly ToyPoint[] = [
  [30, 55],
  [100, 15],
  [100, 95],
  [190, 15],
  [190, 95],
  [260, 55]
];
const TOY_EDGES = [
  [0, 1],
  [0, 2],
  [1, 2],
  [1, 3],
  [2, 3],
  [2, 4],
  [3, 4],
  [3, 5],
  [4, 5]
] as const;

/** Builds the CSR arrays of an undirected graph: every edge is stored once in each direction. */
export function buildToyGraph(): ToyGraph {
  const rows: number[][] = TOY_NODES.map(() => []);
  for (const [from, to] of TOY_EDGES) {
    rows[from]?.push(to);
    rows[to]?.push(from);
  }
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (const row of rows) {
    row.sort((left, right) => left - right);
    for (const neighbor of row) {
      neighbors.push(neighbor);
      weights.push(1 / row.length);
    }
    offsets.push(neighbors.length);
  }
  return {nodes: TOY_NODES, edges: TOY_EDGES, offsets, neighbors, weights};
}

export const TOY_GRAPH = buildToyGraph();
