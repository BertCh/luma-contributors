// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MAXIMUM_RESIDENT_JOIN_COLUMNS, type ResidentJoinProvenance} from './resident-join-plan';
import {
  getResidentSpatialJoinPointTransform,
  normalizeResidentSpatialJoinCrs
} from './resident-spatial-join-reproject';
import type {ResidentSpatialJoinProvenance} from './resident-spatial-join-plan';

/** Saved lineage of a resident Join or Spatial Join result; the result table itself is the data. */
export type ResidentJoinLineage = ResidentJoinProvenance | ResidentSpatialJoinProvenance;

/** Validate and detach saved lineage. The restored table is a snapshot; parents may be absent. */
export function parseResidentJoinLineage(value: unknown): ResidentJoinLineage {
  const record = readRecord(value);
  if (record['schemaVersion'] !== 1) reject('Unsupported saved Join policy.');
  const kind = record['kind'];
  if (kind !== 'join' && kind !== 'spatial-join') reject('Unsupported saved Join kind.');
  const columns = record['columns'];
  const fields = record['fields'];
  if (
    !Array.isArray(columns) ||
    !Array.isArray(fields) ||
    columns.length < 1 ||
    columns.length > MAXIMUM_RESIDENT_JOIN_COLUMNS ||
    fields.length !== columns.length
  )
    reject('Invalid saved Join columns.');
  const names = new Set<string>();
  const parsedFields = fields.map((field, index) => {
    const parsed = readRecord(field);
    const name = readText(parsed['name']);
    if (names.has(name) || name !== readRecord(columns[index])['name'])
      reject('Saved Join output schema is inconsistent.');
    names.add(name);
    return {
      name,
      type: readText(parsed['type']),
      analyzerType: readText(parsed['analyzerType'])
    };
  });
  const common = {
    schemaVersion: 1 as const,
    left: readSource(record['left']),
    right: readSource(record['right']),
    operationId: readText(record['operationId']),
    resultId: readText(record['resultId']),
    resultLabel: readText(record['resultLabel']),
    fields: parsedFields
  };
  if (kind === 'join') {
    const joinType = record['joinType'];
    const keyPolicy = record['keyPolicy'];
    if (
      (joinType !== 'LEFT' && joinType !== 'INNER' && joinType !== 'FULL') ||
      (keyPolicy !== 'string-dictionary-exact-null-and-empty-unmatched' &&
        keyPolicy !== 'numeric-string-form-integral-float32-null-unmatched') ||
      record['rowOrder'] !== 'left-source-order-right-source-order-then-unmatched-right'
    )
      reject('Unsupported saved Join policy.');
    return {
      ...common,
      kind,
      joinType,
      leftKey: readText(record['leftKey']),
      rightKey: readText(record['rightKey']),
      columns: columns.map(column => {
        const parsed = readRecord(column);
        const side = parsed['side'];
        if (side !== 'left' && side !== 'right') reject('Invalid saved Join column.');
        return {side, column: readText(parsed['column']), name: readText(parsed['name'])};
      }),
      keyPolicy,
      rowOrder: 'left-source-order-right-source-order-then-unmatched-right',
      ...(record['duplicateRightMatches'] === 'first' || record['duplicateRightMatches'] === 'last'
        ? {duplicateRightMatches: record['duplicateRightMatches']}
        : {}),
      ...(typeof record['rightColumnSuffix'] === 'string'
        ? {rightColumnSuffix: record['rightColumnSuffix']}
        : {})
    };
  }
  const predicate = record['predicate'];
  const boundaryPolicy = record['boundaryPolicy'];
  const leftGeo = readRecord(record['leftGeo']);
  const rightGeo = readRecord(record['rightGeo']);
  if (
    (predicate !== 'intersects' && predicate !== 'within') ||
    (boundaryPolicy !== 'closed' && boundaryPolicy !== 'open') ||
    record['numericalPolicy'] !== 'float32' ||
    record['rowOrder'] !== 'left-source-order'
  )
    reject('Unsupported saved Spatial Join policy.');
  return {
    ...common,
    kind,
    predicate,
    leftGeo: {latField: readText(leftGeo['latField']), lngField: readText(leftGeo['lngField'])},
    rightGeo: {fieldName: readText(rightGeo['fieldName'])},
    columns: columns.map(column => {
      const parsed = readRecord(column);
      const columnKind = parsed['kind'];
      const technique = parsed['technique'];
      if (
        (columnKind !== 'left' && columnKind !== 'count' && columnKind !== 'aggregate') ||
        (technique !== null &&
          technique !== 'count' &&
          technique !== 'sum' &&
          technique !== 'average' &&
          technique !== 'maximum' &&
          technique !== 'minimum' &&
          technique !== 'merge')
      )
        reject('Invalid saved Spatial Join column.');
      return {
        kind: columnKind,
        column: parsed['column'] === null ? null : readText(parsed['column']),
        technique,
        name: readText(parsed['name'])
      };
    }),
    numericalPolicy: 'float32',
    ...(record['crs'] === undefined ? {} : {crs: parseSpatialJoinCrs(record['crs'])}),
    boundaryPolicy,
    rowOrder: 'left-source-order'
  };
}

/** Native `derivedDataset` metadata for the pinned Kepler dataset-ops lineage. */
export function getResidentJoinDerivedDataset(lineage: ResidentJoinLineage) {
  return {
    type: lineage.kind === 'join' ? 'join' : 'spatialJoin',
    sourceDataIds: [lineage.left.datasetId, lineage.right.datasetId],
    operationId: lineage.operationId
  };
}

function readSource(value: unknown): {datasetId: string; revision: number} {
  const record = readRecord(value);
  const revision = record['revision'];
  if (typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 0)
    reject('Invalid saved Join source.');
  return {datasetId: readText(record['datasetId']), revision};
}
function readRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    reject('Invalid saved Join metadata.');
  return value as Record<string, unknown>;
}
function readText(value: unknown): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 4096)
    reject('Invalid saved Join text.');
  return value;
}
function reject(message: string): never {
  throw new Error(message);
}

function parseSpatialJoinCrs(value: unknown): NonNullable<ResidentSpatialJoinProvenance['crs']> {
  const record = readRecord(value);
  const points = normalizeResidentSpatialJoinCrs(readText(record['points']));
  const polygons = normalizeResidentSpatialJoinCrs(readText(record['polygons']));
  if (!points || !polygons) reject('Unsupported saved Spatial Join CRS.');
  return {points, polygons, pointTransform: getResidentSpatialJoinPointTransform(points, polygons)};
}
