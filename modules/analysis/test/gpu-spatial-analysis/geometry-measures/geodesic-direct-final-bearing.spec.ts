// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// FX-17: the Vincenty direct final bearing is `atan2(sinAlpha, -northTerm)`. `-northTerm` is -0.0
// when `northTerm` is +0.0, which needs `sinU1 * sinSigma == cosU1 * cosSigma * cosAlpha1`
// exactly: an equator origin (sinU1 = 0) with a due east or west bearing (cosAlpha1 = 0 if the GPU
// `cos` rounds it so), and at the poles (cosU1 = 0). The test GPU's `atan2(y, -0.0)` then reports
// the opposite half-turn. Each row is compared with the binary64 Vincenty oracle.

import {type Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getVincentyDirect} from './vincenty-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const ROWS: {origin: [number, number]; bearing: number; distance: number}[] = [
  {origin: [0, 0], bearing: 90, distance: 1000},
  {origin: [0, 0], bearing: 90, distance: 1000000},
  {origin: [0, 0], bearing: -90, distance: 1000},
  {origin: [0, 0], bearing: 270, distance: 1000000},
  {origin: [-0, -0], bearing: 90, distance: 5000},
  {origin: [-0, -0], bearing: -90, distance: 5000},
  {origin: [30, 0], bearing: 0, distance: 1000},
  {origin: [30, 0], bearing: 180, distance: 1000},
  {origin: [0, 0], bearing: 45, distance: 100000},
  {origin: [0, 0], bearing: -135, distance: 100000},
  {origin: [10, 89.99], bearing: 90, distance: 1000},
  {origin: [10, -89.99], bearing: -90, distance: 1000}
];

function getBearingError(actual: number, expected: number): number {
  const difference = Math.abs(actual - expected) % 360;
  return Math.min(difference, 360 - difference);
}

/**
 * Rows whose final bearing is missing, extra, nonfinite or more than 1e-3 degrees (wrapped) from
 * the finite expectation. `getBearingError` is NaN for a NaN bearing, so the bound is written to
 * fail on NaN (TOOL-109).
 */
function getFinalBearingFailures(
  actual: ArrayLike<number>,
  expected: number[],
  labels: string[]
): string[] {
  const failures: string[] = [];
  if (actual.length !== expected.length) {
    failures.push(`length ${actual.length} != ${expected.length}`);
  }
  expected.forEach((target, index) => {
    const value = actual[index];
    if (
      !Number.isFinite(value) ||
      !Number.isFinite(target) ||
      !(getBearingError(value, target) <= 1e-3)
    ) {
      failures.push(`row ${index} (${labels[index]}): GPU ${value}, CPU ${target}`);
    }
  });
  return failures;
}

it('GPUGeodesicDestination Vincenty final bearing is correct for due east and west bearings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const origins = new Float32Array(ROWS.flatMap(row => row.origin));
  const bearings = new Float32Array(ROWS.map(row => row.bearing));
  const distances = new Float32Array(ROWS.map(row => row.distance));
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'geodesic-direct-final-bearing'});
  const destinations = track(createOutputBuffer(device, 2 * ROWS.length));
  const finalBearings = track(createOutputBuffer(device, ROWS.length));
  graph.add(
    new GPUGeodesicDestination({
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal',
        units: 'meters'
      },
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        ROWS.length
      ),
      bearings: importGraphBuffer(
        graph,
        'bearings',
        track(createInputBuffer(device, bearings)),
        'float32',
        ROWS.length
      ),
      distances: importGraphBuffer(
        graph,
        'distances',
        track(createInputBuffer(device, distances)),
        'float32',
        ROWS.length
      ),
      output: {
        destinations: importGraphBuffer(
          graph,
          'o-destinations',
          destinations,
          'float32x2',
          ROWS.length
        ),
        finalBearings: importGraphBuffer(graph, 'o-final', finalBearings, 'float32', ROWS.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const actualFinal = await readFloat32(finalBearings, ROWS.length);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  const expectedFinal = ROWS.map(
    (_, index) =>
      getVincentyDirect(
        [origins[2 * index], origins[2 * index + 1]],
        bearings[index],
        distances[index]
      ).finalBearing
  );
  const labels = ROWS.map(row => `${row.origin}, ${row.bearing}, ${row.distance}`);
  const failures = getFinalBearingFailures(actualFinal, expectedFinal, labels);
  expect(failures).toEqual([]);
});
