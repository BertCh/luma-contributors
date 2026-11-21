// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeMobilityIndicatorsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {CPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
const input = (longitudes: number[]) => ({
  timestamps: Float32Array.of(0, 1),
  staypoints: {
    tracks: Uint32Array.of(0, 0),
    centroids: Float32Array.of(longitudes[0], 0, longitudes[1], 0),
    startRows: Uint32Array.of(0, 1),
    durations: Float32Array.of(1, 1),
    locationIds: Uint32Array.of(0, 1),
    count: Uint32Array.of(2)
  }
});
const spatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  sphereRadius: 6371000
} as const;
it('offers spherical gyration while retaining the coordinate mean', () => {
  for (const [gyrationCenter, expected] of [
    ['coordinate-mean', 19903892],
    ['spherical-mean', 111194.9296875]
  ] as const) {
    const result = computeMobilityIndicatorsOnCPU(input([179, -179]), {
      trackCount: 1,
      spatialContext,
      gyrationCenter
    });
    // R * theta: 6371000 * 179*pi/180, or 6371000*pi/180; rounded to f32.
    expect(Number.isFinite(result.radiusOfGyration[0])).toBe(true);
    expect(result.radiusOfGyration[0]).toBeCloseTo(expected, -2);
  }
});
it('reports an undefined spherical center for cancelling antipodes', () => {
  const result = computeMobilityIndicatorsOnCPU(input([0, 180]), {
    trackCount: 1,
    spatialContext,
    gyrationCenter: 'spherical-mean'
  });
  expect(Number.isNaN(result.radiusOfGyration[0])).toBe(true);
});

for (const longitudes of [
  [-1, 1],
  [179, -179]
]) {
  it(`omitted geographic gyration center follows the short arc at ${longitudes}`, () => {
    const result = computeMobilityIndicatorsOnCPU(input(longitudes), {
      trackCount: 1,
      spatialContext
    });
    // Two equal masses separated by 2 degrees: RMS radius = R*pi/180,
    // exactly half their equatorial jump. f32(6371000*pi/180)=111194.9296875.
    expect(Number.isFinite(result.radiusOfGyration[0])).toBe(true);
    expect(result.radiusOfGyration[0]).toBe(111194.9296875);
    expect(result.radiusOfGyration[0]).toBe(result.jumpLengths[1] / 2);
  });
  it(`CPU named form defaults to spherical gyration at ${longitudes}`, async () => {
    const operation = new CPUMobilityIndicatorsOperation({
      trackCount: 1,
      rowCount: 2,
      staypointCapacity: 2,
      timestamps: 'times',
      spatialContext,
      staypoints: {
        tracks: 'tracks',
        centroids: 'centroids',
        startRows: 'starts',
        durations: 'durations',
        locationIds: 'locations',
        status: {count: 'active'}
      },
      output: {radiusOfGyration: 'radius'}
    });
    const values = input(longitudes);
    const column = (
      format: 'float32' | 'float32x2' | 'uint32',
      packed: Float32Array | Uint32Array
    ) => {
      const length = packed.length / (format === 'float32x2' ? 2 : 1);
      return {format, length, chunks: [{values: packed, length}], pack: () => packed};
    };
    const result = await operation.execute({
      inputs: {
        timestamps: column('float32', values.timestamps),
        stayTracks: column('uint32', values.staypoints.tracks),
        centroids: column('float32x2', values.staypoints.centroids),
        stayStartRows: column('uint32', values.staypoints.startRows),
        durations: column('float32', values.staypoints.durations),
        locationIds: column('uint32', values.staypoints.locationIds),
        stayCount: column('uint32', values.staypoints.count)
      }
    });
    expect(Array.from(result['radius'] as Float32Array)).toEqual([111194.9296875]);
  });
}

it('omitted spherical center remains undefined for balanced antipodes', () => {
  const result = computeMobilityIndicatorsOnCPU(input([0, 180]), {trackCount: 1, spatialContext});
  expect(Number.isNaN(result.radiusOfGyration[0])).toBe(true);
});

it('omitted planar center retains the coordinate mean', () => {
  const result = computeMobilityIndicatorsOnCPU(input([179, -179]), {trackCount: 1});
  expect(result.radiusOfGyration[0]).toBe(179);
});
