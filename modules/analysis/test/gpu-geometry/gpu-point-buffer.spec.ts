// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '@luma-contributors/analysis/geospatial-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

const EARTH_RADIUS_METERS = 6371008.8;

/** Float64 spherical destination formula, the oracle for the float32 offsets. */
function getDestinationOffset(
  longitude: number,
  latitude: number,
  radius: number,
  bearing: number
): [number, number] {
  const toRadians = Math.PI / 180;
  const delta = radius / EARTH_RADIUS_METERS;
  const phi = latitude * toRadians;
  const nextLatitude = Math.asin(
    Math.sin(phi) * Math.cos(delta) + Math.cos(phi) * Math.sin(delta) * Math.cos(bearing)
  );
  const nextLongitude =
    longitude * toRadians +
    Math.atan2(
      Math.sin(bearing) * Math.sin(delta) * Math.cos(phi),
      Math.cos(delta) - Math.sin(phi) * Math.sin(nextLatitude)
    );
  return [nextLongitude / toRadians - longitude, (nextLatitude - phi) / toRadians];
}

it('GPUPointBuffer geographic offsets match the spherical destination formula', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const centers = [
    [-87.65, 41.85],
    [10.2, -33.4],
    [150, 0.5]
  ];
  const verticesPerRing = 12;
  const radius = 100;
  const graph = new GPUCommandGraph(device, {id: 'point-buffer-test'});
  const pointBuffer = createInputBuffer(device, Float32Array.from(centers.flat()));
  const parameterBuffer = createInputBuffer(device, getGPUPointBufferParameterValues(radius));
  const offsetBuffer = createOutputBuffer(device, centers.length * verticesPerRing * 2);
  graph.add(
    new GPUPointBuffer({
      points: importGraphBuffer(graph, 'points', pointBuffer, 'float32x2', centers.length),
      parameters: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 2),
      offsets: importGraphBuffer(
        graph,
        'offsets',
        offsetBuffer,
        'float32x2',
        centers.length * verticesPerRing
      ),
      verticesPerRing,
      coordinateSystem: 'geographic'
    })
  );
  submitGraph(device, graph.compile(), undefined);
  const offsets = await readFloat32(offsetBuffer, centers.length * verticesPerRing * 2);
  centers.forEach(([longitude, latitude], center) => {
    for (let vertex = 0; vertex < verticesPerRing; vertex++) {
      const [expectedLongitude, expectedLatitude] = getDestinationOffset(
        longitude,
        latitude,
        radius,
        (2 * Math.PI * vertex) / verticesPerRing
      );
      const index = 2 * (center * verticesPerRing + vertex);
      // 100 m is about 9e-4 degrees; the offsets keep ~1e-4 relative precision or better.
      expect(Math.abs(offsets[index] - expectedLongitude)).toBeLessThan(2e-7);
      expect(Math.abs(offsets[index + 1] - expectedLatitude)).toBeLessThan(2e-7);
    }
  });
  for (const buffer of [pointBuffer, parameterBuffer, offsetBuffer]) buffer.destroy();
});

/** Great-circle angle in radians between two longitude/latitude points in degrees. */
function getAngularDistance(
  first: readonly [number, number],
  second: readonly [number, number]
): number {
  const toRadians = Math.PI / 180;
  const firstLatitude = first[1] * toRadians;
  const secondLatitude = second[1] * toRadians;
  const longitudeDifference = (second[0] - first[0]) * toRadians;
  const dot =
    Math.sin(firstLatitude) * Math.sin(secondLatitude) +
    Math.cos(firstLatitude) * Math.cos(secondLatitude) * Math.cos(longitudeDifference);
  return Math.acos(Math.min(1, Math.max(-1, dot)));
}

it('GPUPointBuffer geographic rings stay valid at the poles and across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const centers = [
    [0, 90],
    [37, -90],
    [12, 89.9999],
    [-100, 89.5],
    [179.9999, 0],
    [-179.9999, 30]
  ];
  const verticesPerRing = 16;
  const radius = 1000;
  const expectedAngle = radius / EARTH_RADIUS_METERS;
  const graph = new GPUCommandGraph(device, {id: 'point-buffer-pole-test'});
  const pointBuffer = createInputBuffer(device, Float32Array.from(centers.flat()));
  const parameterBuffer = createInputBuffer(device, getGPUPointBufferParameterValues(radius));
  const offsetBuffer = createOutputBuffer(device, centers.length * verticesPerRing * 2);
  graph.add(
    new GPUPointBuffer({
      points: importGraphBuffer(graph, 'points', pointBuffer, 'float32x2', centers.length),
      parameters: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 2),
      offsets: importGraphBuffer(
        graph,
        'offsets',
        offsetBuffer,
        'float32x2',
        centers.length * verticesPerRing
      ),
      verticesPerRing,
      coordinateSystem: 'geographic'
    })
  );
  submitGraph(device, graph.compile(), undefined);
  const offsets = await readFloat32(offsetBuffer, centers.length * verticesPerRing * 2);
  centers.forEach(([longitude, latitude], center) => {
    const longitudes = new Set<number>();
    for (let vertex = 0; vertex < verticesPerRing; vertex++) {
      const index = 2 * (center * verticesPerRing + vertex);
      const longitudeOffset = offsets[index];
      const latitudeOffset = offsets[index + 1];
      expect(Number.isFinite(longitudeOffset) && Number.isFinite(latitudeOffset)).toBe(true);
      expect(Math.abs(longitudeOffset)).toBeLessThanOrEqual(180);
      // The consumer adds the offsets to the float64 center and wraps the longitude.
      const absoluteLatitude = latitude + latitudeOffset;
      expect(absoluteLatitude).toBeLessThanOrEqual(90);
      expect(absoluteLatitude).toBeGreaterThanOrEqual(-90);
      const absoluteLongitude = ((((longitude + longitudeOffset + 180) % 360) + 360) % 360) - 180;
      const angle = getAngularDistance(
        [longitude, latitude],
        [absoluteLongitude, absoluteLatitude]
      );
      expect(Math.abs(angle / expectedAngle - 1), `center ${center} vertex ${vertex}`).toBeLessThan(
        0.02
      );
      longitudes.add(Math.round(absoluteLongitude * 100));
    }
    if (Math.abs(latitude) >= 89.9) {
      // Pole rings spread over all longitudes instead of collapsing to one arbitrary value.
      expect(longitudes.size).toBeGreaterThan(verticesPerRing / 2);
    }
  });
  for (const buffer of [pointBuffer, parameterBuffer, offsetBuffer]) buffer.destroy();
});

it('getGPUPointBufferParameterValues rejects non-positive radii', () => {
  expect(() => getGPUPointBufferParameterValues(0)).toThrow();
  expect(() => getGPUPointBufferParameterValues(-5)).toThrow();
  expect(() => getGPUPointBufferParameterValues(Number.NaN)).toThrow();
  expect(getGPUPointBufferParameterValues(5)[0]).toBe(5);
});
