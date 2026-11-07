// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {makeGeoJSONTripGPUVectors, prepareGeoJSONTrips} from '@luma.gl/engine/arrow';

export function createTripRuntimeFixture(device: Device) {
  const trip = {
    type: 'LineString',
    coordinates: [
      [-74, 40, 0, 1000],
      [-73, 41, 1, 1100]
    ]
  };
  const trips = makeGeoJSONTripGPUVectors(
    device,
    prepareGeoJSONTrips(
      [
        {id: 'left', rows: [null, trip]},
        {id: 'empty', rows: []},
        {id: 'right', rows: [trip, null]}
      ],
      {timestampEncoding: 'unix-millisecond'}
    )
  );
  const vector = <Format extends 'uint32' | 'float32' | 'unorm8x4'>(
    format: Format,
    values: Uint32Array | Float32Array
  ): GPUVector<Format> =>
    new GPUVector({
      type: 'data',
      name: `trip-source-${format}`,
      format,
      ownsData: true,
      data: [values.slice(0, 1), values.slice(1)].map(
        data =>
          new GPUData({
            format,
            length: data.length,
            ownsBuffer: true,
            buffer: device.createBuffer({
              data,
              usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
            })
          })
      )
    });
  const selectionMask = vector('uint32', new Uint32Array([1, 0, 1, 0]));
  const colors = vector(
    'unorm8x4',
    new Uint32Array([0xff030201, 0xff060504, 0xff090807, 0xff0c0b0a])
  );
  const widths = vector('float32', new Float32Array([10, 20, 30, 40]));
  return {
    trips,
    selectionMask,
    colors,
    widths,
    destroy() {
      trips.destroy();
      selectionMask.destroy();
      colors.destroy();
      widths.destroy();
    }
  };
}
