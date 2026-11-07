// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {makeTabularTripGPUVectors, prepareTabularTrips} from '@luma.gl/engine/arrow';

/** Rows 0-3 in batch 0 and rows 4-5 in batch 1: trip a = rows 0,4; trip b = rows 1,2,5. */
export function createTabularTripRuntimeFixture(device: Device, mask: readonly number[]) {
  const row = (id: string, time: number) => [id, time, -74 + time / 10, 40] as const;
  const rows = [
    [row('a', 1000), row('b', 1000), row('b', 1100), row('c', 1)],
    [row('a', 1100), row('b', 1200)]
  ];
  const trips = makeTabularTripGPUVectors(
    device,
    prepareTabularTrips(
      rows.map((batchRows, index) => ({
        id: `batch-${index}`,
        tripIds: batchRows.map(value => value[0]),
        timestamps: batchRows.map(value => value[1]),
        longitudes: batchRows.map(value => value[2]),
        latitudes: batchRows.map(value => value[3])
      })),
      {timestampEncoding: 'unix-millisecond'}
    )
  );
  const selectionMask = new GPUVector({
    type: 'data',
    name: 'tabular-trip-source-mask',
    format: 'uint32',
    ownsData: true,
    data: [mask.slice(0, 4), mask.slice(4)].map(
      values =>
        new GPUData({
          format: 'uint32',
          length: values.length,
          ownsBuffer: true,
          buffer: device.createBuffer({
            data: new Uint32Array(values),
            usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
          })
        })
    )
  });
  return {
    trips,
    selectionMask,
    destroy() {
      trips.destroy();
      selectionMask.destroy();
    }
  };
}
