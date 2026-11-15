// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUVector, VertexList} from '@luma.gl/graph/gpu-data';
import type {PreparedKeplerGeoJSONTrips} from './geojson-trips';
import {
  allocateStorageBuffer,
  makeBatchedGPUVector,
  makePropertyGPUVectors,
  makeSourceRowResolver,
  type StorageBufferValues
} from './gpu-upload-utils';
import type {GeometryGPUProperties} from './geometry-gpu-vectors';

export type KeplerGeoJSONTripGPUVectors = {
  source: PreparedKeplerGeoJSONTrips;
  getPath: GPUVector<VertexList<'float32x3'>>;
  getTimestamps: GPUVector<VertexList<'float32'>>;
  sourceRowIds: GPUVector<'uint32'>;
  properties?: Pick<GeometryGPUProperties, 'lineColor' | 'lineWidth'>;
  /** Explicit grouping index for tabular trips: per-chunk member offsets and global source rows. */
  membership?: {memberOffsets: GPUVector<'uint32'>; memberRows: GPUVector<'uint32'>};
  gpuByteLength: number;
  resolveSourceRow(primitiveIndex: number): number;
  destroy(): void;
};

/** Uploads one path/time/source-row chunk per source batch, without packing or submitting work. */
export function makeKeplerGeoJSONTripGPUVectors(
  device: Device,
  source: PreparedKeplerGeoJSONTrips
): KeplerGeoJSONTripGPUVectors {
  const buffers: Buffer[] = [];
  const allocateBuffer = (name: string, values: StorageBufferValues): Buffer => {
    const buffer = allocateStorageBuffer(device, name, values);
    buffers.push(buffer);
    return buffer;
  };
  try {
    const getPath = makeBatchedGPUVector(
      'trip-paths',
      'vertex-list<float32x3>',
      source.batches.map((batch, index) => ({
        buffer: allocateBuffer(`trip-paths-${index}`, batch.positions),
        length: batch.sourceRows.length,
        valueLength: batch.timestamps.length,
        valueOffsets: batch.valueOffsets
      }))
    );
    const getTimestamps = makeBatchedGPUVector(
      'trip-timestamps',
      'vertex-list<float32>',
      source.batches.map((batch, index) => ({
        buffer: allocateBuffer(`trip-times-${index}`, batch.timestamps),
        length: batch.sourceRows.length,
        valueLength: batch.timestamps.length,
        valueOffsets: batch.valueOffsets
      }))
    );
    const sourceRowIds = makeBatchedGPUVector(
      'trip-source-rows',
      'uint32',
      source.batches.map((batch, index) => ({
        buffer: allocateBuffer(`trip-source-rows-${index}`, batch.sourceRows),
        length: batch.sourceRows.length
      }))
    );
    const uploadProperty = <Format extends 'float32' | 'unorm8x4'>(
      name: 'lineColor' | 'lineWidth',
      format: Format
    ) =>
      source.sourceStyleProperties.includes(name)
        ? makePropertyGPUVectors({
            namePrefix: 'trip',
            name,
            format,
            batches: source.batches.map(batch => ({
              rowCount: batch.rowCount,
              channel: batch.properties![name]!
            })),
            allocateBuffer
          })
        : undefined;
    const properties: Pick<GeometryGPUProperties, 'lineColor' | 'lineWidth'> | undefined = source
      .sourceStyleProperties.length
      ? {
          lineColor: uploadProperty('lineColor', 'unorm8x4'),
          lineWidth: uploadProperty('lineWidth', 'float32')
        }
      : undefined;
    const resolveSourceRow = makeSourceRowResolver(source.batches.map(batch => batch.sourceRows));
    let destroyed = false;
    return {
      source,
      properties,
      getPath,
      getTimestamps,
      sourceRowIds,
      gpuByteLength: buffers.reduce((total, buffer) => total + buffer.byteLength, 0),
      resolveSourceRow,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        getPath.destroy();
        getTimestamps.destroy();
        sourceRowIds.destroy();
        for (const channel of Object.values(properties ?? {})) {
          channel?.values.destroy();
          channel?.validity.destroy();
        }
        for (const buffer of buffers) buffer.destroy();
      }
    };
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    throw error;
  }
}
