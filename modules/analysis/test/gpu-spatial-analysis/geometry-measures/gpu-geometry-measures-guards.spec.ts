// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  ANTARCTICA_NATURAL_EARTH,
  ARCTIC_RING,
  SPIRAL_WIDER_THAN_360,
  GREAT_CIRCLE_60_DEGREE_EDGE
} from '../lon-lat-polygons/fixtures';
import {
  getFixtureRings,
  getChamberlainDuquetteArea,
  getSphericalTriangleArea
} from './geometry-measures-oracle';
import {expect, it} from 'vitest';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createFlatGeometry,
  getMeasuresGeometryFields,
  type NestedFeatures
} from './geometry-measures-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle values: modules/experimental scratch script
// <scratchpad>/fix/work/E/E4-measures-lines/guards_oracle.py (pyproj 3.8.0, numpy 2.5.4).
/** pyproj.Geod(ellps='WGS84').polygon_area_perimeter of the Fiji-like ring below, square meters. */
const FIJI_GEODESIC_AREA = 176366280779.48474;
/** Unit-vector mean of two points at (179, 10) and (-179, 10): longitude 180, this latitude. */
const TWO_POINT_CENTROID_LATITUDE = 10.001492526984078;
/** Unit-vector mean of (170, 40) x1, (-170, 45) x2, (175, -20) x1 (weights = point counts). */
const WEIGHTED_CENTROID = [-179.43038247860034, 29.082693050024577];

// Longitude 182 is -178: the second edge steps 5 degrees east across the antimeridian.
const FIJI: NestedFeatures[number] = [
  [
    [177, -19],
    [-178, -19],
    [-178, -16],
    [177, -16]
  ]
];
// Four vertices around the north pole: wrapped longitude steps sum to +360 degrees.
const POLE_RING: NestedFeatures[number] = [
  [
    [0, 80],
    [90, 80],
    [180, 80],
    [-90, 80]
  ]
];
// An edge spanning exactly 180 degrees of longitude has no unique shortest side.
const HALF_WORLD_EDGE: NestedFeatures[number] = [
  [
    [0, 10],
    [180, 10],
    [180, 20],
    [0, 20]
  ]
];
const SMALL_BOX: NestedFeatures[number] = [
  [
    [10, 20],
    [11, 20],
    [11, 21],
    [10, 21]
  ]
];

function wrapDelta(delta: number): number {
  return delta - 360 * Math.round(delta / 360);
}

async function measure(
  features: NestedFeatures,
  options: {
    geometryType: 'polygons' | 'points';
    metric: 'great-circle' | 'ellipsoidal';
    groupIds?: number[];
    groupCount?: number;
    large?: boolean;
    edgeModel?: 'linear' | 'great-circle';
  }
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const flat = createFlatGeometry(features);
  const featureCount = features.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'geometry-measures-guards'});
  const rows = options.groupCount ?? featureCount;
  const areas = track(createOutputBuffer(device, featureCount));
  const centroids = track(createOutputBuffer(device, 2 * featureCount));
  const invalid = track(createOutputBuffer(device, 1));
  const groupCentroids = track(createOutputBuffer(device, 2 * rows));
  const groupAreas = track(createOutputBuffer(device, rows));
  const isPolygon = options.geometryType === 'polygons';
  graph.add(
    new GPUGeometryMeasures({
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: options.metric,
        edgeModel: options.edgeModel,
        units: 'meters'
      },
      ...getMeasuresGeometryFields(
        flat,
        options.geometryType,
        importGraphBuffer(
          graph,
          'positions',
          track(createInputBuffer(device, flat.positions)),
          'float32x2',
          flat.positions.length / 2
        ),
        (name, values) =>
          importGraphBuffer(
            graph,
            name,
            track(createInputBuffer(device, values)),
            'uint32',
            values.length
          )
      ),
      // Force the 64-lane cooperative kernel for every ring when requested.
      cooperativeRingRows: options.large ? 1 : undefined,
      output: {
        ...(isPolygon
          ? {areas: importGraphBuffer(graph, 'areas', areas, 'float32', featureCount)}
          : {}),
        centroids: importGraphBuffer(graph, 'centroids', centroids, 'float32x2', featureCount),
        status: {invalidCount: importGraphBuffer(graph, 'invalid', invalid, 'uint32', 1)}
      },
      ...(options.groupIds
        ? {
            groupIds: importGraphBuffer(
              graph,
              'group-ids',
              track(createInputBuffer(device, new Uint32Array(options.groupIds))),
              'uint32',
              featureCount
            ),
            groupCount: options.groupCount,
            groupOutput: {
              ...(isPolygon
                ? {areas: importGraphBuffer(graph, 'group-areas', groupAreas, 'float32', rows)}
                : {}),
              centroids: importGraphBuffer(
                graph,
                'group-centroids',
                groupCentroids,
                'float32x2',
                rows
              )
            }
          }
        : {})
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    areas: isPolygon ? await readFloat32(areas, featureCount) : [],
    centroids: await readFloat32(centroids, 2 * featureCount),
    invalidCount: (await readUint32(invalid, 1))[0],
    groupCentroids: await readFloat32(groupCentroids, 2 * rows),
    groupAreas: await readFloat32(groupAreas, rows)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

for (const large of [false, true]) {
  it(`GPUGeometryMeasures wraps antimeridian rings and flags pole and half-world rings (${
    large ? 'cooperative' : 'serial'
  })`, async () => {
    const result = await measure([FIJI, POLE_RING, HALF_WORLD_EDGE, SMALL_BOX], {
      geometryType: 'polygons',
      metric: 'ellipsoidal',
      large
    });
    if (!result) {
      return;
    }
    // An edge with a longitude difference of 355 degrees is wrapped to 5: the area is the
    // geodesic area pyproj reports (f32 positions, so 1e-4 relative).
    expect(Math.abs(result.areas[0] - FIJI_GEODESIC_AREA) / FIJI_GEODESIC_AREA).toBeLessThan(1e-4);
    // Pole-enclosing and half-world rings report NaN area and centroid, and are counted.
    for (const feature of [1, 2]) {
      expect(result.areas[feature], `area ${feature}`).toBeNaN();
      expect(result.centroids[2 * feature], `centroid x ${feature}`).toBeNaN();
      expect(result.centroids[2 * feature + 1], `centroid y ${feature}`).toBeNaN();
    }
    expect(result.areas[3]).toBeGreaterThan(0);
    expect(Number.isFinite(result.areas[3])).toBe(true);
    expect(result.invalidCount).toBe(2);
  });
}

it('GPUGeometryMeasures group centroids are spherical means across the antimeridian', async () => {
  const twoPoints = await measure([[[[179, 10]]], [[[-179, 10]]]], {
    geometryType: 'points',
    metric: 'great-circle',
    groupIds: [0, 0],
    groupCount: 1
  });
  if (!twoPoints) {
    return;
  }
  expect(Math.abs(wrapDelta(twoPoints.groupCentroids[0] - 180))).toBeLessThan(1e-4);
  expect(twoPoints.groupCentroids[1]).toBeCloseTo(TWO_POINT_CENTROID_LATITUDE, 4);

  // Group weights are vertex counts for point sets.
  const weighted = await measure(
    [
      [[[170, 40]]],
      [
        [
          [-170, 45],
          [-170, 45]
        ]
      ],
      [[[175, -20]]]
    ],
    {geometryType: 'points', metric: 'great-circle', groupIds: [0, 0, 0], groupCount: 1}
  );
  expect(Math.abs(wrapDelta(weighted!.groupCentroids[0] - WEIGHTED_CENTROID[0]))).toBeLessThan(
    1e-4
  );
  expect(weighted!.groupCentroids[1]).toBeCloseTo(WEIGHTED_CENTROID[1], 4);
});

for (const large of [false, true]) {
  it(`GPUGeometryMeasures pole edges and empty rings (${large ? 'cooperative' : 'serial'})`, async () => {
    const antarctica = getFixtureRings(ANTARCTICA_NATURAL_EARTH);
    const result = await measure(
      [antarctica, getFixtureRings(ARCTIC_RING), getFixtureRings(SPIRAL_WIDER_THAN_360), [[]]],
      {
        geometryType: 'polygons',
        metric: 'great-circle',
        large
      }
    );
    const expected = getChamberlainDuquetteArea(antarctica[0], 6371007.180918475);
    expect(Math.abs(result.areas[0] / expected - 1)).toBeLessThan(1e-5);
    expect(result.areas[1]).toBeNaN();
    expect(result.areas[2]).toBeNaN();
    expect(result.areas[3]).toBe(0);
    expect(result.invalidCount).toBe(2);
  });

  it(`GPUGeometryMeasures great-circle triangle area (${large ? 'cooperative' : 'serial'})`, async () => {
    const triangle = getFixtureRings(GREAT_CIRCLE_60_DEGREE_EDGE);
    const result = await measure([triangle, [[]]], {
      geometryType: 'polygons',
      metric: 'great-circle',
      edgeModel: 'great-circle',
      large
    });
    const expected = getSphericalTriangleArea(triangle[0], 6371008.8);
    expect(Math.abs(result.areas[0] / expected - 1)).toBeLessThan(1e-6);
    expect(result.areas[1]).toBe(0);
    expect(result.invalidCount).toBe(0);
  });
}

// Math correctness review 2026-11-10, geometry-measures-01. Rings a few hundred meters across:
// the f32 unit-vector fan the kernel used to sum returned 60,920 m2 for the first shell (true
// 10,008 m2) and a negative area for the second.
const SMALL_GREAT_CIRCLE_RINGS: NestedFeatures = [
  [
    [
      [10.3, 45.2],
      [10.301276, 45.2],
      [10.301276, 45.2009],
      [10.3, 45.2009]
    ]
  ],
  [
    [
      [-122.4, 37.7],
      [-122.398863, 37.7],
      [-122.398863, 37.7009],
      [-122.4, 37.7009]
    ]
  ],
  [
    [
      [10.3, 45.2],
      [10.31276, 45.2],
      [10.31276, 45.209],
      [10.3, 45.209]
    ]
  ],
  [
    [
      [30, 80],
      [30.00518, 80],
      [30.00518, 80.0009],
      [30, 80.0009]
    ]
  ],
  // Shell with a clockwise hole: the hole's sign comes from its own (small) ring area.
  [
    [
      [-122.4, 37.7],
      [-122.398863, 37.7],
      [-122.398863, 37.7009],
      [-122.4, 37.7009]
    ],
    [
      [-122.3997, 37.7002],
      [-122.3997, 37.7006],
      [-122.3992, 37.7006],
      [-122.3992, 37.7002]
    ]
  ]
];

/** Binary64 l'Huilier area of a convex quadrilateral ring whose vertices are rounded to f32. */
function getSmallRingArea(ring: number[][]): number {
  const rounded = ring.map(vertex => [Math.fround(vertex[0]), Math.fround(vertex[1])]);
  return (
    getSphericalTriangleArea([rounded[0], rounded[1], rounded[2]], 6371008.8) +
    getSphericalTriangleArea([rounded[0], rounded[2], rounded[3]], 6371008.8)
  );
}

for (const large of [false, true]) {
  it(`GPUGeometryMeasures great-circle area of rings a hundred meters across (${
    large ? 'cooperative' : 'serial'
  })`, async () => {
    const result = await measure(SMALL_GREAT_CIRCLE_RINGS, {
      geometryType: 'polygons',
      metric: 'great-circle',
      edgeModel: 'great-circle',
      large
    });
    SMALL_GREAT_CIRCLE_RINGS.forEach((feature, index) => {
      const expected =
        getSmallRingArea(feature[0]) - (feature[1] ? getSmallRingArea(feature[1]) : 0);
      expect(Math.abs(result.areas[index] / expected - 1), `feature ${index}`).toBeLessThan(1e-5);
    });
    // Hand check of the first shell: 100.0 m by 100.08 m at latitude 45.2.
    expect(result.areas[0]).toBeGreaterThan(10000);
    expect(result.areas[0]).toBeLessThan(10020);
    expect(result.invalidCount).toBe(0);
  });
}

// A concave ring (reflex at the third vertex) that crosses the antimeridian and the equator. The
// diagonal from the first to the third vertex is interior, so the area is the sum of two
// spherical triangles (binary64 l'Huilier), independent of the kernel's per-edge sum.
const CONCAVE_ANTIMERIDIAN_RING = [
  [170, -10],
  [-170, 0],
  [178, 8],
  [172, 30]
];

for (const large of [false, true]) {
  it(`GPUGeometryMeasures great-circle area of a concave ring across the antimeridian (${
    large ? 'cooperative' : 'serial'
  })`, async () => {
    const result = await measure(
      [[CONCAVE_ANTIMERIDIAN_RING], [[...CONCAVE_ANTIMERIDIAN_RING].reverse()]],
      {geometryType: 'polygons', metric: 'great-circle', edgeModel: 'great-circle', large}
    );
    const [first, second, third, fourth] = CONCAVE_ANTIMERIDIAN_RING;
    const expected =
      getSphericalTriangleArea([first, second, third], 6371008.8) +
      getSphericalTriangleArea([first, third, fourth], 6371008.8);
    expect(expected).toBeGreaterThan(2.9e12);
    expect(Math.abs(result.areas[0] / expected - 1)).toBeLessThan(2e-6);
    // The clockwise copy has the same unsigned area.
    expect(Math.abs(result.areas[1] / expected - 1)).toBeLessThan(2e-6);
    expect(result.invalidCount).toBe(0);
  });
}

// Math correctness review 2026-11-10, geometry-measures-06. The centroid of a longitude/latitude
// box lies at the mean authalic sine of its two parallels; the expected latitudes are the exact
// inverse of the closed-form authalic latitude (binary64 Newton iteration, scratch script
// authalic-check.mjs). The forward-series coefficient 17/360 the kernel used moved them by 4.5e-5
// degrees (5.05 m).
const AUTHALIC_CENTROID_BOXES: {south: number; north: number; latitude: number}[] = [
  {south: 22.4, north: 22.6, latitude: 22.499964679776852},
  {south: 67.4, north: 67.6, latitude: 67.49979015842966},
  {south: -22.6, north: -22.4, latitude: -22.499964679776838},
  {south: 10, north: 35, latitude: 21.950942713733028}
];

it('GPUGeometryMeasures ellipsoidal polygon centroids invert the authalic latitude', async () => {
  const result = await measure(
    AUTHALIC_CENTROID_BOXES.map(box => [
      [
        [5, box.south],
        [6, box.south],
        [6, box.north],
        [5, box.north]
      ]
    ]),
    {geometryType: 'polygons', metric: 'ellipsoidal'}
  );
  AUTHALIC_CENTROID_BOXES.forEach((box, index) => {
    expect(result.centroids[2 * index], `longitude ${index}`).toBeCloseTo(5.5, 5);
    // The kernel takes asin of an f32 authalic sine, so the latitude carries the sine's rounding
    // divided by cos(latitude): 8e-6 degrees (0.9 m) at the equator, 2.1e-5 at latitude 67.5.
    expect(
      Math.abs(result.centroids[2 * index + 1] - box.latitude),
      `latitude ${index}`
    ).toBeLessThan(8e-6 / Math.cos((box.latitude * Math.PI) / 180));
  });
});
