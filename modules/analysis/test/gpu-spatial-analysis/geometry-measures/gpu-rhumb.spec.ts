// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const EARTH_RADIUS = 6371008.8;
const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

/** f64 closed forms of the sphere rhumb line (the formulas of turf `rhumb*` and geo `Rhumb`). */
function getRhumbPair(a: number[], b: number[]) {
  const phi1 = a[1] * D2R;
  const phi2 = b[1] * D2R;
  let deltaLambda = (b[0] - a[0]) * D2R;
  if (Math.abs(deltaLambda) > Math.PI) {
    deltaLambda += deltaLambda > 0 ? -2 * Math.PI : 2 * Math.PI;
  }
  const deltaPsi = Math.log(Math.tan(Math.PI / 4 + phi2 / 2) / Math.tan(Math.PI / 4 + phi1 / 2));
  const deltaPhi = phi2 - phi1;
  const q = Math.abs(deltaPsi) > 1e-12 ? deltaPhi / deltaPsi : Math.cos(phi1);
  // Midpoint by distance: latitude is linear in distance along a loxodrome and longitude is linear
  // in the Mercator latitude psi, so the longitude fraction is (psi(mid) - psi1) / (psi2 - psi1).
  // An east-west line has the plain mean longitude.
  const phiMiddle = 0.5 * (phi1 + phi2);
  const fraction =
    Math.abs(deltaPsi) > 1e-12
      ? Math.log(Math.tan(Math.PI / 4 + phiMiddle / 2) / Math.tan(Math.PI / 4 + phi1 / 2)) /
        deltaPsi
      : 0.5;
  return {
    distance: Math.hypot(deltaPhi, q * deltaLambda) * EARTH_RADIUS,
    bearing: Math.atan2(deltaLambda, deltaPsi) * R2D,
    midpoint: [a[0] + fraction * (deltaLambda * R2D), 0.5 * (a[1] + b[1])]
  };
}

/**
 * Rhumb midpoints checked by hand (math correctness review 2026-11-10, geometry-measures-02).
 * `(0, 0) -> (90, 60)`: psi(30) = ln(tan 60) = ln(sqrt 3) = 0.5493061, psi(60) = ln(tan 75) =
 * ln(2 + sqrt 3) = 1.3169579, so the longitude is 90 * 0.5493061 / 1.3169579 = 37.5392. The plain
 * mean longitude 45, which the kernel used to return, is 718 km from the loxodrome's midpoint.
 * `(-74, 40.7) -> (-0.1, 51.5)` (New York to London): psi = 0.7789409, 0.9087903, 1.0520657, so
 * the longitude is -74 + 73.9 * (0.9087903 - 0.7789409) / (1.0520657 - 0.7789409) = -38.8664
 * (the mean longitude -37.05 is 140 km east of it).
 * `(170, -10) -> (-170, 50)` crosses the antimeridian going east: psi(-10) = -0.1754258, psi(20) =
 * 0.3563785, psi(50) = 1.0106832, so the longitude is 170 + 20 * 0.5318043 / 1.1861090 = 178.9672.
 */
const HAND_MIDPOINTS: [string, number[], number[], number[]][] = [
  ['quarter turn', [0, 0], [90, 60], [37.5392, 30]],
  ['New York to London', [-74, 40.7], [-0.1, 51.5], [-38.8664, 46.1]],
  ['antimeridian', [170, -10], [-170, 50], [178.9672, 20]],
  ['east-west', [10, 60], [30, 60], [20, 60]],
  ['north-south', [10, -20], [10, 50], [10, 15]],
  // Fix review fr-geometry-03: lines that reach very high latitudes, where the kernel used to
  // divide a half difference of isometric latitude by a whole one clamped at atanh(0.999999) and
  // returned 10.93 and 18.81. By hand: psi(89.95) = ln(cot 0.025) = ln(2291.83) = 7.7371 and
  // psi(44.975) = ln(tan 67.4875) = 0.8805, so the longitude is 90 * 0.8805 / 7.7371 = 10.245;
  // psi(60) = 1.3170, psi(74.995) = ln(tan 82.4975) = 2.0271 and psi(89.99) = ln(cot 0.005) =
  // 9.3466, so the longitude is 10 + 90 * (2.0271 - 1.3170) / (9.3466 - 1.3170) = 17.96.
  ['equator to 89.95', [0, 0], [90, 89.95], [10.2453, 44.975]],
  ['60 to 89.99', [10, 60], [100, 89.99], [17.9616, 74.995]],
  ['pole to pole', [0, -88], [120, 88], [60, 0]]
];

/**
 * Latitudes of rhumb lines that are symmetric across the equator, `(0, -latitude)` to
 * `(100, latitude)` and back. Their midpoint is on the equator at the mean longitude exactly: the
 * two halves have the same isometric-latitude difference.
 */
const SYMMETRIC_LATITUDES = [20, 60, 80, 85, 88, 89.99];

function getRhumbDestination(origin: number[], bearing: number, distance: number): number[] {
  const delta = distance / EARTH_RADIUS;
  const phi1 = origin[1] * D2R;
  const theta = bearing * D2R;
  const deltaPhi = delta * Math.cos(theta);
  let phi2 = phi1 + deltaPhi;
  if (Math.abs(phi2) > Math.PI / 2) {
    phi2 = phi2 > 0 ? Math.PI - phi2 : -Math.PI - phi2;
  }
  const deltaPsi = Math.log(Math.tan(phi2 / 2 + Math.PI / 4) / Math.tan(phi1 / 2 + Math.PI / 4));
  const q = Math.abs(deltaPsi) > 1e-11 ? deltaPhi / deltaPsi : Math.cos(phi1);
  return [origin[0] + ((delta * Math.sin(theta)) / q) * R2D, phi2 * R2D];
}

/**
 * Pinned from turf (`@turf/rhumb-distance`, `rhumb-bearing`, `rhumb-destination` 7.x, f64, meters,
 * radius 6371008.8); generator in scratchpad build/J/gen.mjs.
 */
const TURF_PAIRS: [string, number[], number[], number, number][] = [
  ['1 m', [2.35, 48.85], [2.350013, 48.850001], 0.9576870745754993, 83.33246619922852],
  ['1 km', [-73.98, 40.75], [-73.97, 40.752], 871.2242351558756, 75.21090863073431],
  ['100 km', [139.7, 35.7], [140.5, 36.4], 105976.75200217181, 42.737699719311934],
  ['1,000 km', [-0.13, 51.5], [12.5, 41.9], 1435306.4704436501, 138.04967075079367],
  ['transpacific', [151.2, -33.9], [-118.2, 34], 12108362.161459818, 51.42430062401968],
  ['antimeridian', [179.5, 10], [-179.5, 12], 247730.22859042592, 26.141194384294977],
  ['east-west', [10, 60], [30, 60], 1111950.8023353291, 90],
  ['north-south', [10, -20], [10, 50], 7783655.616347303, 0],
  ['polar', [0, 89.9], [179, 89.8], 51336.28927988376, 102.50949020230757]
];

const TURF_DESTINATIONS: [string, number[], number, number, number[]][] = [
  ['100 km NE', [2, 48], 45, 100000, [2.9562865493012396, 48.63591552764876]],
  ['1,000 km E', [-30, 50], 90, 1000000, [-16.00905882172958, 50]],
  ['500 km S', [100, -10], 180, 500000, [100, -14.49660181862269]],
  ['3,000 km W', [170, 20], 270, 3000000, [141.28889775768187, 19.999999999999993]],
  ['1 km NNW', [-73.98, 40.75], 340, 1000, [-73.98406044675147, 40.75845084709515]],
  ['zero', [10, 10], 77, 0, [10, 10]]
];

function getBearingError(actual: number, expected: number): number {
  const difference = Math.abs(actual - expected) % 360;
  return Math.min(difference, 360 - difference);
}

it('rhumb f64 closed form reproduces the pinned turf values', () => {
  for (const [name, a, b, distance, bearing] of TURF_PAIRS) {
    const actual = getRhumbPair(a, b);
    expect(Math.abs(actual.distance - distance), name).toBeLessThan(1e-6 + 1e-9 * distance);
    expect(getBearingError(actual.bearing, bearing), name).toBeLessThan(1e-9);
  }
  for (const [name, origin, bearing, distance, expected] of TURF_DESTINATIONS) {
    const actual = getRhumbDestination(origin, bearing, distance);
    expect(Math.abs(actual[0] - expected[0]), name).toBeLessThan(1e-9);
    expect(Math.abs(actual[1] - expected[1]), name).toBeLessThan(1e-9);
  }
});

it('GPUGeodesicPairs rhumb matches the f64 closed form (pinned to turf)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = TURF_PAIRS.length;
  const origins = new Float32Array(TURF_PAIRS.flatMap(([, a]) => a));
  const targets = new Float32Array(TURF_PAIRS.flatMap(([, , b]) => b));
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'rhumb-pairs'});
  const outputs = {
    distances: track(createOutputBuffer(device, count)),
    initialBearings: track(createOutputBuffer(device, count)),
    finalBearings: track(createOutputBuffer(device, count)),
    midpoints: track(createOutputBuffer(device, 2 * count))
  };
  graph.add(
    new GPUGeodesicPairs({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'},
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        count
      ),
      targets: importGraphBuffer(
        graph,
        'targets',
        track(createInputBuffer(device, targets)),
        'float32x2',
        count
      ),

      output: {
        distances: importGraphBuffer(graph, 'o-d', outputs.distances, 'float32', count),
        initialBearings: importGraphBuffer(graph, 'o-i', outputs.initialBearings, 'float32', count),
        finalBearings: importGraphBuffer(graph, 'o-f', outputs.finalBearings, 'float32', count),
        midpoints: importGraphBuffer(graph, 'o-m', outputs.midpoints, 'float32x2', count)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const distances = await readFloat32(outputs.distances, count);
  const initial = await readFloat32(outputs.initialBearings, count);
  const final = await readFloat32(outputs.finalBearings, count);
  const midpoints = await readFloat32(outputs.midpoints, 2 * count);
  const report: string[] = [];
  TURF_PAIRS.forEach(([name], row) => {
    // The f64 reference sees the same f32-rounded inputs the GPU does.
    const a = [origins[2 * row], origins[2 * row + 1]];
    const b = [targets[2 * row], targets[2 * row + 1]];
    const expected = getRhumbPair(a, b);
    expect(final[row], `${name} final`).toBe(initial[row]);
    const error = Math.abs(distances[row] - expected.distance);
    report.push(`${name}: ${error.toExponential(2)} m`);
    expect(error, name).toBeLessThanOrEqual(1e-3 + 3e-7 * expected.distance);
    expect(getBearingError(initial[row], expected.bearing), `${name} bearing`).toBeLessThan(
      name === '1 m' ? 1e-2 : 1e-4
    );
    expect(Math.abs(midpoints[2 * row] - expected.midpoint[0]), `${name} mid x`).toBeLessThan(1e-4);
    expect(Math.abs(midpoints[2 * row + 1] - expected.midpoint[1]), `${name} mid y`).toBeLessThan(
      1e-4
    );
  });
  // Same-latitude and same-meridian edges hit the east-west limit and the zero-longitude case.
  expect(Math.abs(distances[6] - 1111950.8023353291)).toBeLessThan(5);
  expect(initial[6]).toBeCloseTo(90, 4);
  expect(initial[7]).toBeCloseTo(0, 4);
  console.log(`GPUGeodesicPairs rhumb distance errors vs f64 closed form: ${report.join(', ')}`);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
});

it('GPUGeodesicPairs rhumb midpoints halve the rhumb distance on the rhumb line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const symmetricPairs = SYMMETRIC_LATITUDES.flatMap(latitude => [
    [
      [0, -latitude],
      [100, latitude]
    ],
    [
      [100, latitude],
      [0, -latitude]
    ]
  ]);
  const count = HAND_MIDPOINTS.length + symmetricPairs.length;
  const origins = new Float32Array([
    ...HAND_MIDPOINTS.flatMap(([, a]) => a),
    ...symmetricPairs.flatMap(([a]) => a)
  ]);
  const targets = new Float32Array([
    ...HAND_MIDPOINTS.flatMap(([, , b]) => b),
    ...symmetricPairs.flatMap(([, b]) => b)
  ]);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'rhumb-midpoints'});
  const output = track(createOutputBuffer(device, 2 * count));
  graph.add(
    new GPUGeodesicPairs({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'},
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        count
      ),
      targets: importGraphBuffer(
        graph,
        'targets',
        track(createInputBuffer(device, targets)),
        'float32x2',
        count
      ),
      output: {midpoints: importGraphBuffer(graph, 'o-m', output, 'float32x2', count)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const midpoints = await readFloat32(output, 2 * count);
  HAND_MIDPOINTS.forEach(([name, , , expected], row) => {
    const a = [origins[2 * row], origins[2 * row + 1]];
    const b = [targets[2 * row], targets[2 * row + 1]];
    const midpoint = [midpoints[2 * row], midpoints[2 * row + 1]];
    expect(Math.abs(midpoint[0] - expected[0]), `${name} longitude`).toBeLessThan(1e-4);
    expect(Math.abs(midpoint[1] - expected[1]), `${name} latitude`).toBeLessThan(1e-4);
    // Independent of the midpoint formula: the turf-pinned rhumb distance and bearing put the
    // point halfway along the line and on it.
    const whole = getRhumbPair(a, b);
    const firstHalf = getRhumbPair(a, midpoint);
    const secondHalf = getRhumbPair(midpoint, b);
    expect(Math.abs(firstHalf.distance - 0.5 * whole.distance), `${name} first half`).toBeLessThan(
      10 + 2e-6 * whole.distance
    );
    expect(
      Math.abs(secondHalf.distance - 0.5 * whole.distance),
      `${name} second half`
    ).toBeLessThan(10 + 2e-6 * whole.distance);
    expect(getBearingError(firstHalf.bearing, whole.bearing), `${name} bearing`).toBeLessThan(1e-3);
  });
  // Symmetric lines: exactly the mean longitude on the equator, in both directions.
  symmetricPairs.forEach((pair, index) => {
    const row = HAND_MIDPOINTS.length + index;
    expect(midpoints[2 * row], `symmetric ${pair[0]} longitude`).toBe(50);
    expect(midpoints[2 * row + 1], `symmetric ${pair[0]} latitude`).toBe(0);
  });
  // The whole line is pinned too at the highest latitude.
  expect(midpoints[2 * (HAND_MIDPOINTS.length - 1)]).toBe(60);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
});

// Math correctness review 2026-11-10, geometry-measures-08. An east-west rhumb line of longitude
// difference d at latitude phi is R cos(phi) d long; at latitude 60 that is R d / 2. The pair is
// 0.0547332763671875 degrees apart across the antimeridian: 6086.071 m at the equator. Wrapping
// after the subtraction, as the kernel used to, loses 1.5e-5 degrees (1.7 m).
it('GPUGeodesicPairs rhumb keeps a short separation across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const origins = new Float32Array([179.94722, 0, -179.99805, 60]);
  const targets = new Float32Array([-179.99805, 0, 179.94722, 60]);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'rhumb-seam'});
  const distanceBuffer = track(createOutputBuffer(device, 2));
  const bearingBuffer = track(createOutputBuffer(device, 2));
  const midpointBuffer = track(createOutputBuffer(device, 4));
  graph.add(
    new GPUGeodesicPairs({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'},
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        2
      ),
      targets: importGraphBuffer(
        graph,
        'targets',
        track(createInputBuffer(device, targets)),
        'float32x2',
        2
      ),
      output: {
        distances: importGraphBuffer(graph, 'o-d', distanceBuffer, 'float32', 2),
        initialBearings: importGraphBuffer(graph, 'o-i', bearingBuffer, 'float32', 2),
        midpoints: importGraphBuffer(graph, 'o-m', midpointBuffer, 'float32x2', 2)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const distances = await readFloat32(distanceBuffer, 2);
  const bearings = await readFloat32(bearingBuffer, 2);
  const midpoints = await readFloat32(midpointBuffer, 4);
  const separation = ((360 - origins[0] + targets[0]) * Math.PI) / 180;
  expect(Math.abs(distances[0] - EARTH_RADIUS * separation)).toBeLessThan(0.01);
  expect(Math.abs(distances[1] - 0.5 * EARTH_RADIUS * separation)).toBeLessThan(0.01);
  expect(getBearingError(bearings[0], 90)).toBeLessThan(1e-4);
  expect(getBearingError(bearings[1], -90)).toBeLessThan(1e-4);
  // The midpoint is continuous with the origin: half the separation east, then west.
  const halfSeparationDegrees = 0.5 * (360 - origins[0] + targets[0]);
  expect(Math.abs(midpoints[0] - (origins[0] + halfSeparationDegrees))).toBeLessThan(2e-5);
  expect(Math.abs(midpoints[2] - (origins[2] - halfSeparationDegrees))).toBeLessThan(2e-5);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
});

it('GPUGeodesicDestination rhumb matches the f64 closed form (pinned to turf)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = [
    ...TURF_DESTINATIONS.map(([, origin, bearing, distance]) => ({origin, bearing, distance})),
    // Overshoots the north pole: reflected back across it.
    {origin: [20, 85], bearing: 0, distance: 1000000}
  ];
  const count = rows.length;
  const origins = new Float32Array(rows.flatMap(row => row.origin));
  const bearings = new Float32Array(rows.map(row => row.bearing));
  const distances = new Float32Array(rows.map(row => row.distance));
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'rhumb-destination'});
  const destinations = track(createOutputBuffer(device, 2 * count));
  const finalBearings = track(createOutputBuffer(device, count));
  graph.add(
    new GPUGeodesicDestination({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'},
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        count
      ),
      bearings: importGraphBuffer(
        graph,
        'bearings',
        track(createInputBuffer(device, bearings)),
        'float32',
        count
      ),
      distances: importGraphBuffer(
        graph,
        'distances',
        track(createInputBuffer(device, distances)),
        'float32',
        count
      ),

      output: {
        destinations: importGraphBuffer(graph, 'o-dest', destinations, 'float32x2', count),
        finalBearings: importGraphBuffer(graph, 'o-final', finalBearings, 'float32', count)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const actual = await readFloat32(destinations, 2 * count);
  const actualFinal = await readFloat32(finalBearings, count);
  const report: string[] = [];
  rows.forEach((row, index) => {
    const origin = [origins[2 * index], origins[2 * index + 1]];
    const expected = getRhumbDestination(origin, row.bearing, row.distance);
    const errorLng = Math.abs(actual[2 * index] - expected[0]);
    const errorLat = Math.abs(actual[2 * index + 1] - expected[1]);
    report.push(`#${index}: ${errorLng.toExponential(1)}/${errorLat.toExponential(1)} deg`);
    // Near the pole the longitude is ill-conditioned (q -> 0), so allow a looser bound there.
    expect(errorLng, `row ${index} lng`).toBeLessThan(index === count - 1 ? 0.05 : 2e-4);
    expect(errorLat, `row ${index} lat`).toBeLessThan(2e-4);
    if (index < count - 1) {
      expect(getBearingError(actualFinal[index], row.bearing)).toBeLessThan(1e-4);
    }
  });
  // North-pole overshoot flips the final bearing to southbound.
  expect(getBearingError(actualFinal[count - 1], 180)).toBeLessThan(1e-4);
  console.log(`GPUGeodesicDestination rhumb errors vs f64 closed form: ${report.join(', ')}`);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
});
