// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Math review luma-official-12: the package has two point-in-polygon classifiers. The inline exact
// one of `GPUPointInPolygonJoin` (with a `boundaryRule`), and published `GPUPairwisePointInPolygon`
// inside `GPUSpatialPredicateJoin` (rows it calls `uncertain` fall back to the fast predicate
// kernel). This spec runs both on the same boundary rows:
// - `boundaryRule: 'closed'` must keep the smallest polygon row that `intersects` the point;
// - `boundaryRule: 'open'` must keep the smallest polygon row the point is `within`.
// The expected owners are written by hand from the drawn geometry.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE as NO_FEATURE,
  GPUPointInPolygonJoin,
  type GPUPointInPolygonBoundaryRule
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildPolygonFeatureArrays} from './spatial-join-oracle';
import type {OracleFeature, OraclePoint} from './spatial-predicate-oracle';
import {runScaleJoin} from './spatial-relate-scale-harness';

type Owner = number | null;

/** One multipolygon feature: polygons, then rings (shell first), then vertices. */
type PolygonFeature = OraclePoint[][][];

type Scene = {
  name: string;
  features: PolygonFeature[];
  points: [number, number][];
  /** Smallest feature row whose closed polygon holds the point. */
  closedOwners: Owner[];
  /** Smallest feature row whose interior holds the point. */
  openOwners: Owner[];
};

const ring = (...vertices: [number, number][]): [number, number][] => vertices;
const box = (x0: number, y0: number, x1: number, y1: number): PolygonFeature => [
  [ring([x0, y0], [x1, y0], [x1, y1], [x0, y1])]
];

/** A 2 by 2 coverage of side-2 squares at `(x, y)`: rows 0 and 1 below, rows 2 and 3 above. */
function getSquaresScene(name: string, x: number, y: number): Scene {
  return {
    name,
    features: [
      box(x, y, x + 2, y + 2),
      box(x + 2, y, x + 4, y + 2),
      box(x, y + 2, x + 2, y + 4),
      box(x + 2, y + 2, x + 4, y + 4)
    ],
    points: (
      [
        [2, 1], // shared edge of rows 0 and 1
        [1, 2], // shared edge of rows 0 and 2
        [2, 2], // the vertex shared by all four
        [2, 3], // shared edge of rows 2 and 3
        [3, 2], // shared edge of rows 1 and 3
        [0, 1], // outer edges and corners
        [1, 0],
        [0, 0],
        [4, 1],
        [1, 4],
        [4, 4],
        [4, 0],
        [0, 4],
        [1, 1], // interiors
        [3, 3],
        [5, 5] // outside
      ] as [number, number][]
    ).map(([pointX, pointY]): [number, number] => [x + pointX, y + pointY]),
    closedOwners: [0, 0, 0, 2, 1, 0, 0, 0, 1, 2, 3, 1, 2, 0, 3, null],
    openOwners: [
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      0,
      3,
      null
    ]
  };
}

const SCENES: Scene[] = [
  getSquaresScene('squares', 0, 0),
  // The same coverage at projected-metre magnitudes (one float32 step is 0.5 at the northing).
  getSquaresScene('squares at projected metres', 500000, 5000000),
  {
    // Two triangles that share the oblique edge (0, 0)-(8, 4), the line y = x / 2.
    name: 'oblique shared edge',
    features: [[[ring([0, 0], [8, 4], [0, 8])]], [[ring([0, 0], [8, 0], [8, 4])]]],
    points: [
      [2, 1], // on the shared edge
      [4, 2],
      [6, 3],
      [8, 4], // shared vertex
      [4, 6], // on row 0's edge (8, 4)-(0, 8)
      [8, 2], // on row 1's vertical edge
      [4, 0], // on row 1's bottom edge
      [2, 1.0000001], // one float32 step above the shared edge: inside row 0
      [2, 0.99999994], // one step below: inside row 1
      [1, 4], // interior of row 0
      [6, 1], // interior of row 1
      [9, 9] // outside
    ],
    closedOwners: [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, null],
    openOwners: [null, null, null, null, null, null, null, 0, 1, 0, 1, null]
  },
  {
    // Row 0 has a hole that touches nothing; row 1 fills the hole. Row 2 carries a second ring of
    // two vertices, which bounds no area: the published classifier reports such a row as
    // uncertain, the inline one skips the ring.
    name: 'hole, hole filler and a degenerate ring',
    features: [
      [[ring([0, 0], [8, 0], [8, 8], [0, 8]), ring([2, 2], [2, 6], [6, 6], [6, 2])]],
      [[ring([2, 2], [6, 2], [6, 6], [2, 6])]],
      [[ring([10, 0], [14, 0], [14, 4], [10, 4]), ring([11, 1], [12, 2])]]
    ],
    points: [
      [4, 4], // inside the hole: only row 1
      [2, 4], // on the hole boundary: rows 0 and 1
      [6, 6], // a hole vertex
      [1, 1], // interior of row 0
      [0, 4], // outer edge of row 0
      [12, 3], // interior of row 2
      [11.5, 1.5], // on the degenerate ring of row 2, which is not a boundary
      [14, 2], // edge of row 2
      [9, 2] // outside
    ],
    closedOwners: [1, 0, 0, 0, 0, 2, 2, 2, null],
    openOwners: [1, null, null, 0, null, 2, 2, null, null]
  }
];

/** Owner per point from `GPUPointInPolygonJoin` (feature row, or `null`). */
async function joinPoints(
  device: Device,
  scene: Scene,
  boundaryRule: GPUPointInPolygonBoundaryRule
): Promise<{owners: Owner[]; uncertain: number}> {
  const graph = new GPUCommandGraph(device, {id: 'pip-agreement'});
  const arrays = buildPolygonFeatureArrays(scene.features);
  const buffers: Buffer[] = [];
  const input = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const pointCount = scene.points.length;
  const pointFeatureIds = output(pointCount);
  const uncertainCount = output(1);
  const overflow = output(1);
  const offsets = (name: string, data: Uint32Array) =>
    importGraphBuffer(graph, name, input(data), 'uint32', data.length);
  graph.add(
    new GPUPointInPolygonJoin({
      points: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'points',
          input(Float32Array.from(scene.points.flat())),
          'float32x2',
          pointCount
        )
      },
      polygons: {
        kind: 'polygons',
        positions: importGraphBuffer(
          graph,
          'polygon-positions',
          input(arrays.polygonPositions),
          'float32x2',
          arrays.polygonPositions.length / 2
        ),
        featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
        polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
        ringOffsets: offsets('ring-offsets', arrays.ringOffsets)
      },
      candidateCapacity: pointCount * scene.features.length,
      boundaryRule,
      output: {
        pointFeatureIds: importGraphBuffer(
          graph,
          'point-feature-ids',
          pointFeatureIds,
          'uint32',
          pointCount
        ),
        status: {
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          uncertainCount: importGraphBuffer(graph, 'uncertain-count', uncertainCount, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const ids = await readUint32(pointFeatureIds, pointCount);
  const [uncertain] = await readUint32(uncertainCount, 1);
  expect((await readUint32(overflow, 1))[0]).toBe(0);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {owners: ids.map(id => (id === NO_FEATURE ? null : id)), uncertain};
}

/** Smallest matching polygon row per point from `GPUSpatialPredicateJoin`. */
async function joinWithPredicate(
  device: Device,
  scene: Scene,
  predicate: 'intersects' | 'within'
): Promise<Owner[]> {
  const points: OracleFeature[] = scene.points.map(vertex => ({kind: 'points', vertex}));
  const polygons: OracleFeature[] = scene.features.map(feature => ({
    kind: 'polygons',
    polygons: feature
  }));
  const result = await runScaleJoin(device, points, polygons, predicate, {});
  expect(result.overflow).toBe(0);
  const owners: Owner[] = scene.points.map(() => null);
  for (const pair of result.pairs) {
    const [pointRow, polygonRow] = pair.split(',').map(Number);
    const current = owners[pointRow];
    owners[pointRow] = current === null ? polygonRow : Math.min(current, polygonRow);
  }
  return owners;
}

for (const scene of SCENES) {
  it(`the two point-in-polygon classifiers agree on boundary rows: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const closed = await joinPoints(device, scene, 'closed');
    const open = await joinPoints(device, scene, 'open');
    expect(closed.uncertain, 'closed uncertain').toBe(0);
    expect(open.uncertain, 'open uncertain').toBe(0);
    expect(closed.owners, 'GPUPointInPolygonJoin closed').toEqual(scene.closedOwners);
    expect(open.owners, 'GPUPointInPolygonJoin open').toEqual(scene.openOwners);
    expect(
      await joinWithPredicate(device, scene, 'intersects'),
      'GPUSpatialPredicateJoin intersects'
    ).toEqual(scene.closedOwners);
    expect(
      await joinWithPredicate(device, scene, 'within'),
      'GPUSpatialPredicateJoin within'
    ).toEqual(scene.openOwners);
  });
}
