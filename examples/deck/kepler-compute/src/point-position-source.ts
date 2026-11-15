// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import type {PreparedArrowAnalyticsTable} from '@luma-contributors/analysis/analysis-core';
import type {KeplerDatasetRegistry, KeplerMemoryReservation} from '@luma-contributors/kepler';

import type {FlowRoles} from './flow-source';
export type CoordinateRoles = {
  longitude: string;
  latitude: string;
  altitude?: string;
  /** Native Flow layer roles; longitude/latitude are the flow source coordinates. */
  flow?: FlowRoles;
  /** Arc and line target coordinates; longitude/latitude are the connection origins. */
  target?: {longitude: string; latitude: string; altitude?: string};
};
export type PreparedTabularPointPositions = {
  roles: Readonly<CoordinateRoles>;
  batches: readonly {
    sourceRowOffset: number;
    rowCount: number;
    positions: Float32Array;
    positions3D?: Float32Array;
    altitudeValidity?: Uint32Array;
  }[];
  bounds?: [number, number, number, number];
  /** Rows whose longitude/latitude were null or out of range; hidden (NaN position), never fatal. */
  invalidRowCount: number;
  typedStorageByteLength: number;
  gpuByteLength: number;
};
export type TabularPointGPUVectors = {
  positions: GPUVector<'float32x2'>;
  positions3D?: GPUVector<'float32x3'>;
  altitudeValidity?: GPUVector<'uint32'>;
  bounds?: [number, number, number, number];
  readonly gpuByteLength: number;
  destroy(completion: PromiseLike<unknown>): Promise<void>;
};

/** Reserve temporary CPU storage before preparation; original Arrow storage is already retained. */
export function estimateTabularPointPositionBytes(
  sourceChunkLengths: readonly number[],
  hasAltitude: boolean
): {cpuByteLength: number; gpuByteLength: number} {
  let cpuByteLength = 0;
  let gpuByteLength = 0;
  for (const length of sourceChunkLengths) {
    if (!Number.isSafeInteger(length) || length < 0)
      throw new Error('Invalid point source chunk length');
    cpuByteLength += length * (hasAltitude ? 24 : 8);
    gpuByteLength +=
      Math.max(8, length * 8) +
      (hasAltitude ? Math.max(12, length * 12) + Math.max(4, length * 4) : 0);
  }
  if (!Number.isSafeInteger(cpuByteLength) || !Number.isSafeInteger(gpuByteLength))
    throw new Error('Point position capacity exceeds safe byte counts');
  return {cpuByteLength, gpuByteLength};
}

/**
 * Read selected normalized coordinates once at ingestion. Float64 conversion follows the retained
 * explicit Arrow exact/round policy; integer coordinates must also be exactly representable as f32.
 * Native Arrow point conversion renders null altitude at zero without changing the retained source.
 * Nonfinite altitude hides only its 3D point; XY spatial inputs retain every source row.
 * Native row-array points instead omit null altitude: this adapter follows the Arrow publication path.
 */
export function prepareTabularPointPositions(
  prepared: PreparedArrowAnalyticsTable,
  roles: CoordinateRoles
): PreparedTabularPointPositions {
  const columns = getPositionColumns(prepared, roles);
  const estimate = estimateTabularPointPositionBytes(
    prepared.sourceBatches.map(batch => batch.sourceRowCount),
    roles.altitude !== undefined
  );
  let minimumLongitude = Infinity;
  let maximumLongitude = -Infinity;
  let minimumLatitude = Infinity;
  let maximumLatitude = -Infinity;
  let rowCount = 0;
  let invalidRowCount = 0;
  const batches = prepared.sourceBatches.map(batch => {
    if (batch.sourceRowIndexOffset !== rowCount)
      throw new Error('Point source offsets must be contiguous');
    const positions = new Float32Array(batch.sourceRowCount * 2);
    const positions3D = columns.altitude ? new Float32Array(batch.sourceRowCount * 3) : undefined;
    const altitudeValidity = columns.altitude ? new Uint32Array(batch.sourceRowCount) : undefined;
    for (let row = 0; row < batch.sourceRowCount; row++) {
      const sourceRow = batch.sourceRowIndexOffset + row;
      const coordinate = readLongitudeLatitude(columns, sourceRow);
      if (!coordinate) {
        // Native Kepler skips rows with null or invalid coordinates; NaN positions are not drawn.
        invalidRowCount++;
        positions[row * 2] = NaN;
        positions[row * 2 + 1] = NaN;
        if (positions3D && altitudeValidity) {
          positions3D.fill(NaN, row * 3, row * 3 + 3);
          altitudeValidity[row] = 0;
        }
        continue;
      }
      const [longitude, latitude] = coordinate;
      positions[row * 2] = longitude;
      positions[row * 2 + 1] = latitude;
      minimumLongitude = Math.min(minimumLongitude, longitude);
      maximumLongitude = Math.max(maximumLongitude, longitude);
      minimumLatitude = Math.min(minimumLatitude, latitude);
      maximumLatitude = Math.max(maximumLatitude, latitude);
      if (positions3D && altitudeValidity && columns.altitude) {
        const altitude = readAltitude(columns.altitude, sourceRow);
        positions3D[row * 3] = longitude;
        positions3D[row * 3 + 1] = latitude;
        positions3D[row * 3 + 2] = altitude ?? 0;
        altitudeValidity[row] = altitude === null ? 0 : 1;
      }
    }
    rowCount += batch.sourceRowCount;
    return {
      sourceRowOffset: batch.sourceRowIndexOffset,
      rowCount: batch.sourceRowCount,
      positions,
      ...(positions3D ? {positions3D, altitudeValidity} : {})
    };
  });
  return {
    roles: Object.freeze({...roles}),
    batches,
    ...(minimumLongitude <= maximumLongitude
      ? {
          bounds: [minimumLongitude, minimumLatitude, maximumLongitude, maximumLatitude] as [
            number,
            number,
            number,
            number
          ]
        }
      : {}),
    invalidRowCount,
    typedStorageByteLength: estimate.cpuByteLength,
    gpuByteLength: estimate.gpuByteLength
  };
}

/** Replay preflight validates the same policy before any dataset replacement, without output arrays. */
export function validateTabularPointPositions(
  prepared: PreparedArrowAnalyticsTable,
  roles: CoordinateRoles
): void {
  const columns = getPositionColumns(prepared, roles);
  for (const batch of prepared.sourceBatches) {
    for (let row = 0; row < batch.sourceRowCount; row++) {
      const sourceRow = batch.sourceRowIndexOffset + row;
      if (readLongitudeLatitude(columns, sourceRow) && columns.altitude)
        readAltitude(columns.altitude, sourceRow);
    }
  }
}

/** Upload each original chunk once; XY and optional XYZ have independent, explicit ownership. */
export function makeGPUVectorsFromTabularPointPositions(
  device: Device,
  prepared: PreparedTabularPointPositions,
  options: {memoryRegistry?: KeplerDatasetRegistry} = {}
): TabularPointGPUVectors {
  const expected = estimateTabularPointPositionBytes(
    prepared.batches.map(batch => batch.rowCount),
    prepared.roles.altitude !== undefined
  );
  let reservation: KeplerMemoryReservation | undefined;
  const buffers: Buffer[] = [];
  const deviceLoss = device.lost;
  const upload = <Format extends GPUVectorFormat>(
    format: Format,
    values: readonly (Float32Array | Uint32Array)[],
    rowByteLength: number
  ): GPUVector<Format> => {
    const data = values.map((values, index) => {
      const rowCount = prepared.batches[index].rowCount;
      if (values.byteLength !== rowCount * rowByteLength)
        throw new Error('Point position chunks must match original source rows');
      const buffer = device.createBuffer({
        ...(values.byteLength ? {data: values} : {byteLength: rowByteLength}),
        usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC
      });
      buffers.push(buffer);
      return new GPUData({format, buffer, length: rowCount, ownsBuffer: false});
    });
    return new GPUVector({
      type: 'data',
      name: `tabular-point-${format}`,
      format,
      data,
      ownsData: false
    });
  };
  try {
    reservation = options.memoryRegistry?.reserveMemory(expected.gpuByteLength, 'gpu', 'source');
    const positions = upload(
      'float32x2',
      prepared.batches.map(batch => batch.positions),
      8
    );
    const positions3D =
      prepared.roles.altitude === undefined
        ? undefined
        : upload(
            'float32x3',
            prepared.batches.map(batch => {
              if (!batch.positions3D) throw new Error('Missing altitude position chunk');
              return batch.positions3D;
            }),
            12
          );
    const altitudeValidity =
      prepared.roles.altitude === undefined
        ? undefined
        : upload(
            'uint32',
            prepared.batches.map(batch => {
              if (!batch.altitudeValidity) throw new Error('Missing altitude validity chunk');
              return batch.altitudeValidity;
            }),
            4
          );
    const byteLength = buffers.reduce((total, buffer) => total + buffer.byteLength, 0);
    if (byteLength !== expected.gpuByteLength)
      throw new Error('Point upload allocation differs from its admitted plan');
    let destruction: Promise<void> | undefined;
    let released = false;
    return {
      positions,
      positions3D,
      altitudeValidity,
      bounds: prepared.bounds,
      get gpuByteLength() {
        return released ? 0 : byteLength;
      },
      destroy(completion) {
        destruction ??= Promise.race([completion, deviceLoss])
          .finally(() => {
            for (const buffer of buffers) buffer.destroy();
            reservation?.release();
            released = true;
          })
          .then(() => undefined);
        return destruction;
      }
    };
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    reservation?.release();
    throw error;
  }
}

type PositionColumns = {longitude: arrow.Vector; latitude: arrow.Vector; altitude?: arrow.Vector};
function getPositionColumns(
  prepared: PreparedArrowAnalyticsTable,
  roles: CoordinateRoles
): PositionColumns {
  const getColumn = (name: string): arrow.Vector => {
    const column = prepared.table.getChild(name);
    if (
      !name ||
      !column ||
      prepared.dictionaries[name] ||
      !(arrow.DataType.isFloat(column.type) || arrow.DataType.isInt(column.type))
    )
      throw new Error('Coordinate roles require selected normalized numeric scalar fields');
    return column;
  };
  return {
    longitude: getColumn(roles.longitude),
    latitude: getColumn(roles.latitude),
    ...(roles.altitude !== undefined ? {altitude: getColumn(roles.altitude)} : {})
  };
}
function readLongitudeLatitude(columns: PositionColumns, row: number): [number, number] | null {
  const longitude: unknown = columns.longitude.get(row);
  const latitude: unknown = columns.latitude.get(row);
  if (
    typeof longitude !== 'number' ||
    typeof latitude !== 'number' ||
    !Number.isFinite(longitude) ||
    !Number.isFinite(latitude) ||
    longitude < -180 ||
    longitude > 180 ||
    latitude < -90 ||
    latitude > 90
  )
    return null;
  if (Math.fround(longitude) !== longitude || Math.fround(latitude) !== latitude)
    throw new Error('Point coordinates require explicit Float32 normalization');
  return [longitude, latitude];
}
function readAltitude(column: arrow.Vector, row: number): number | null {
  const altitude: unknown = column.get(row);
  if (altitude === null || altitude === undefined) return 0;
  if (typeof altitude !== 'number' || !Number.isFinite(altitude)) return null;
  if (Math.fround(altitude) !== altitude)
    throw new Error(`Altitude at source row ${row} requires explicit Float32 normalization`);
  return altitude;
}
