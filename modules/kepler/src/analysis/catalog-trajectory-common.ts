// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '@luma-contributors/analysis/analysis-core';
import {parseKeplerIsoTimestamp} from '../profile/iso-timestamp';
import type {
  CatalogParameterSpec,
  CatalogParameterValue,
  CatalogRunOutput
} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  appendCatalogRow,
  makeCatalogTable,
  makeCatalogGeometryText,
  type CatalogFeatureCollection,
  type CatalogPosition,
  type CatalogColumnType
} from './catalog-geojson';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

// Conservative UI admission limits: several CPU twins scan pairs; eight recursion radii
// produce at most 16,384 rows. These are host responsiveness limits, not contributor maxima.
export const MAXIMUM_TRAJECTORY_FIXES = 2048;
export const MAXIMUM_TRAJECTORY_TRACKS = 128;
export type TrajectoryValues = Readonly<Record<string, CatalogParameterValue>>;
export type TrajectoryFix = {
  sourceIndex: number;
  vertexIndex: number;
  trackId: string;
  trackIndex: number;
  timestamp: number;
  position: CatalogPosition;
  /** Source altitude is retained separately; numerical analysis remains two dimensional. */
  altitude?: number;
  properties: Record<string, unknown>;
};
export type CatalogTrajectory = {
  fixes: TrajectoryFix[];
  skippedEmptyFeatureCount: number;
  positions: Float32Array;
  timestamps: Uint32Array;
  trackOffsets: Uint32Array;
  trackIds: string[];
  frame: LocalFrame;
};

export function numberParameter(
  name: string,
  label: string,
  defaultValue: number,
  minimum = 0,
  maximum = 1e9,
  unit?: string
): CatalogParameterSpec {
  return {name, label, kind: 'number', defaultValue, minimum, maximum, ...(unit ? {unit} : {})};
}
export function enumParameter(
  name: string,
  label: string,
  options: string[],
  defaultValue = options[0]
): CatalogParameterSpec {
  return {name, label, kind: 'enum', options, defaultValue};
}
export function stringParameter(
  name: string,
  label: string,
  defaultValue: string
): CatalogParameterSpec {
  return {name, label, kind: 'string', defaultValue, maximumLength: 128};
}
export const TRAJECTORY_PARAMETERS: readonly CatalogParameterSpec[] = [
  COORDINATE_SYSTEM_PARAMETER,
  stringParameter('trackProperty', 'Track ID property', 'track_id'),
  stringParameter('timeProperty', 'Timestamp property', 'timestamp'),
  enumParameter('timestampEncoding', 'Input timestamp encoding', [
    'unix-millisecond',
    'unix-second',
    'iso8601'
  ]),
  stringParameter('longitudeProperty', 'Point longitude column', 'longitude'),
  stringParameter('latitudeProperty', 'Point latitude column', 'latitude')
];

function readTimestamp(value: unknown, encoding: CatalogParameterValue): number {
  // Missing zones mean UTC; impossible dates are rejected by the existing strict ISO parser.
  // The parser keeps whole milliseconds, so reject nonzero lower digits before they are lost.
  const submillisecondDigits =
    encoding === 'iso8601' && typeof value === 'string'
      ? value.match(/\.\d{3}(\d+)/)?.[1]
      : undefined;
  const timestamp =
    encoding === 'iso8601' && typeof value === 'string'
      ? parseKeplerIsoTimestamp(value)
      : typeof value === 'number'
        ? value * (encoding === 'unix-second' ? 1000 : 1)
        : NaN;
  const milliseconds = Math.round(timestamp);
  if (
    !Number.isSafeInteger(milliseconds) ||
    Math.abs(timestamp - milliseconds) > 1e-6 ||
    (submillisecondDigits !== undefined && /[1-9]/.test(submillisecondDigits))
  )
    throw new Error(
      'Trajectory timestamps must resolve to safe integer epoch milliseconds; sub-millisecond precision is not supported.'
    );
  return milliseconds;
}

/** Packs in stable track/time order, retaining both native row and path vertex identities. */
export function prepareCatalogTrajectory(
  collection: CatalogFeatureCollection,
  values: TrajectoryValues
): CatalogTrajectory {
  // Inspect only feature metadata before allocating or walking individual fixes. Each limit
  // is validated once for this preparation, outside the row loops and contributor execution.
  let fixCount = 0;
  const inputTrackIds = new Set<string>();
  for (const feature of collection.features) {
    if (!feature.geometry) continue;
    const geometry = feature.geometry;
    fixCount +=
      geometry.type === 'LineString' && Array.isArray(geometry.coordinates)
        ? geometry.coordinates.length
        : 1;
    inputTrackIds.add(
      String(feature.properties?.[values['trackProperty'] as string] ?? feature.id)
    );
  }
  validateCapacity('catalog-trajectory', 'fixCount', fixCount, 1, MAXIMUM_TRAJECTORY_FIXES);
  validateCapacity(
    'catalog-trajectory',
    'trackCount',
    inputTrackIds.size,
    1,
    MAXIMUM_TRAJECTORY_TRACKS
  );
  const groups = new Map<string, TrajectoryFix[]>();
  let skippedEmptyFeatureCount = 0;
  collection.features.forEach((feature, sourceIndex) => {
    if (!feature.geometry) {
      skippedEmptyFeatureCount++;
      return;
    }
    const properties = feature.properties ?? {};
    const identity = properties[values['trackProperty'] as string] ?? feature.id;
    if (
      (typeof identity !== 'string' && typeof identity !== 'number') ||
      identity === '' ||
      (typeof identity === 'number' && !Number.isFinite(identity))
    )
      throw new Error(`Trajectory source row ${sourceIndex} requires a track ID.`);
    const trackId = String(identity);
    const geometry = feature.geometry;
    if (geometry?.type !== 'Point' && geometry?.type !== 'LineString')
      throw new Error(`Trajectory source row ${sourceIndex} must be a Point or timed LineString.`);
    const coordinates = geometry.type === 'Point' ? [geometry.coordinates] : geometry.coordinates;
    if (!Array.isArray(coordinates) || !coordinates.length)
      throw new Error(`Trajectory source row ${sourceIndex} has no fixes.`);
    const track = groups.get(trackId) ?? [];
    groups.set(trackId, track);
    coordinates.forEach((coordinate: unknown, vertexIndex: number) => {
      if (
        !Array.isArray(coordinate) ||
        coordinate.length < 2 ||
        !Number.isFinite(coordinate[0]) ||
        !Number.isFinite(coordinate[1])
      )
        throw new Error(`Trajectory source row ${sourceIndex} has invalid coordinates.`);
      if (
        values['coordinateSystem'] === 'EPSG:4326' &&
        (Math.abs(coordinate[0]) > 180 || Math.abs(coordinate[1]) > 90)
      )
        throw new Error('Trajectory longitude/latitude must be within [-180, 180] / [-90, 90].');
      const altitude: unknown = coordinate[2];
      if (altitude !== undefined && (typeof altitude !== 'number' || !Number.isFinite(altitude)))
        throw new Error(`Trajectory source row ${sourceIndex} has invalid altitude.`);
      const propertyTime = properties[values['timeProperty'] as string];
      const time =
        geometry.type === 'Point'
          ? propertyTime
          : (coordinate[3] ??
            (Array.isArray(propertyTime) ? propertyTime[vertexIndex] : undefined));
      track.push({
        sourceIndex,
        vertexIndex,
        trackId,
        trackIndex: 0,
        timestamp: readTimestamp(time, values['timestampEncoding']),
        position: [coordinate[0], coordinate[1]],
        ...(altitude === undefined ? {} : {altitude: altitude as number}),
        properties
      });
    });
  });
  const fixes: TrajectoryFix[] = [];
  const offsets = [0];
  const trackIds = [...groups.keys()];
  trackIds.forEach((trackId, trackIndex) => {
    const track = groups.get(trackId)!;
    track.sort((first, second) => first.timestamp - second.timestamp);
    track.forEach((fix, index) => {
      if (index && fix.timestamp <= track[index - 1].timestamp)
        throw new Error(
          `Trajectory track ${trackId} requires distinct, strictly increasing timestamps.`
        );
      fix.trackIndex = trackIndex;
      fixes.push(fix);
    });
    offsets.push(fixes.length);
  });
  const frame = makeLocalFrame(
    values['coordinateSystem'] as 'EPSG:4326' | 'planar',
    fixes.map(fix => fix.position),
    {shift: 'center'}
  );
  const positions = Float32Array.from(fixes.flatMap(fix => frame.forward(fix.position)));
  if (!positions.every(Number.isFinite))
    throw new Error('Trajectory cannot be represented in the local Float32 frame.');
  const timestamps = new Uint32Array(fixes.length * 2);
  fixes.forEach((fix, index) => {
    const words = BigInt.asUintN(64, BigInt(fix.timestamp));
    timestamps[index * 2] = Number(words & 0xffffffffn);
    timestamps[index * 2 + 1] = Number(words >> 32n);
  });
  return {
    fixes,
    skippedEmptyFeatureCount,
    positions,
    timestamps,
    trackOffsets: Uint32Array.from(offsets),
    trackIds,
    frame
  };
}

/** Restore source altitude even when a cleaning operation supplies a corrected XY position. */
export function getTrajectoryFixPosition(
  fix: TrajectoryFix,
  position = fix.position
): readonly number[] {
  return fix.altitude === undefined ? position : [...position, fix.altitude];
}

export function makeTrajectoryFixes(
  trajectory: CatalogTrajectory,
  columns: Record<string, CatalogColumnType> = {},
  rows = trajectory.fixes.map((_, index) => index),
  extra: (row: number) => Record<string, string | number | boolean | null> = () => ({}),
  positions?: readonly CatalogPosition[]
) {
  const table = makeCatalogTable(
    'trajectory-fixes',
    'Trajectory fixes',
    {
      source_index: 'int32',
      vertex_index: 'int32',
      fix_index: 'int32',
      output_index: 'int32',
      track_id: 'string',
      timestamp: 'float64',
      original_properties: 'string',
      ...columns,
      geometry: 'string'
    },
    'geometry'
  );
  for (const row of rows) {
    const fix = trajectory.fixes[row];
    appendCatalogRow(table, {
      source_index: fix.sourceIndex,
      vertex_index: fix.vertexIndex,
      fix_index: row,
      output_index: table.rowCount,
      track_id: fix.trackId,
      timestamp: fix.timestamp,
      original_properties: JSON.stringify(fix.properties),
      ...extra(row),
      geometry: makeCatalogGeometryText('Point', getTrajectoryFixPosition(fix, positions?.[row]))
    });
  }
  return table;
}

export function getTrajectoryTrackRows(trajectory: CatalogTrajectory): number[][] {
  return trajectory.trackIds.map((_, index) =>
    Array.from(
      {length: trajectory.trackOffsets[index + 1] - trajectory.trackOffsets[index]},
      (_, offset) => trajectory.trackOffsets[index] + offset
    )
  );
}

/** Four-component coordinates use Kepler's native Trips convention [longitude, latitude, altitude, epoch ms]. */
export function makeTrajectoryTracks(
  trajectory: CatalogTrajectory,
  groups = getTrajectoryTrackRows(trajectory),
  positions?: readonly CatalogPosition[]
) {
  const table = makeCatalogTable(
    'trajectory-tracks',
    'Trajectory tracks',
    {
      part_index: 'int32',
      track_id: 'string',
      source_indices: 'string',
      vertex_indices: 'string',
      start_time: 'float64',
      end_time: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  groups.forEach((rows, partIndex) => {
    if (rows.length < 2) return;
    const first = trajectory.fixes[rows[0]];
    appendCatalogRow(table, {
      part_index: partIndex,
      track_id: first.trackId,
      source_indices: JSON.stringify(rows.map(row => trajectory.fixes[row].sourceIndex)),
      vertex_indices: JSON.stringify(rows.map(row => trajectory.fixes[row].vertexIndex)),
      start_time: first.timestamp,
      end_time: trajectory.fixes[rows[rows.length - 1]].timestamp,
      geometry: makeCatalogGeometryText(
        'LineString',
        rows.map(row => [
          ...(positions?.[row] ?? trajectory.fixes[row].position),
          trajectory.fixes[row].altitude ?? 0,
          trajectory.fixes[row].timestamp
        ])
      )
    });
  });
  return table;
}

export function makeTrajectoryOutput(
  trajectory: CatalogTrajectory,
  tables: CatalogRunOutput['tables'],
  summary: CatalogRunOutput['summary'] = {}
): CatalogRunOutput {
  return {
    tables,
    diagnostics: trajectory.skippedEmptyFeatureCount
      ? [
          {
            code: 'trajectory-empty-geometry',
            severity: 'warning',
            count: trajectory.skippedEmptyFeatureCount,
            message:
              'Null geometries were skipped; retained fixes keep their original source row identities.'
          }
        ]
      : [],
    summary: {fixCount: trajectory.fixes.length, trackCount: trajectory.trackIds.length, ...summary}
  };
}
