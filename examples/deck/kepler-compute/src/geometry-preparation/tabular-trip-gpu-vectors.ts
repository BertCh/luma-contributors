// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {allocateStorageBuffer, makeBatchedGPUVector} from './gpu-upload-utils';
import {
  makeKeplerGeoJSONTripGPUVectors,
  type KeplerGeoJSONTripGPUVectors
} from './trip-gpu-vectors';
import type {PreparedTabularTrips} from './tabular-trips';

/**
 * Uploads trip paths/times/primary rows plus the explicit member-row grouping index, one chunk per
 * source batch. The index is an accounted derived resource; source columns are never repacked.
 */
export function makeKeplerTabularTripGPUVectors(
  device: Device,
  source: PreparedTabularTrips
): KeplerGeoJSONTripGPUVectors {
  const vectors = makeKeplerGeoJSONTripGPUVectors(device, source);
  const buffers: Buffer[] = [];
  try {
    const uploadMembers = (
      name: string,
      select: (batch: PreparedTabularTrips['batches'][number]) => Uint32Array
    ) =>
      makeBatchedGPUVector(
        name,
        'uint32',
        source.batches.map((batch, index) => {
          const values = select(batch);
          const buffer = allocateStorageBuffer(device, `${name}-${index}`, values);
          buffers.push(buffer);
          return {buffer, length: values.length};
        })
      );
    const memberOffsets = uploadMembers('trip-member-offsets', batch => batch.memberOffsets);
    const memberRows = uploadMembers('trip-member-rows', batch => batch.memberRows);
    let destroyed = false;
    return {
      ...vectors,
      membership: {memberOffsets, memberRows},
      gpuByteLength:
        vectors.gpuByteLength + buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0),
      destroy() {
        if (destroyed) return;
        destroyed = true;
        vectors.destroy();
        memberOffsets.destroy();
        memberRows.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    };
  } catch (error) {
    vectors.destroy();
    for (const buffer of buffers) buffer.destroy();
    throw error;
  }
}
