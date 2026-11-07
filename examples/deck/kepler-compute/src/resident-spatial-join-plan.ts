// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {planSpatialJoinSemantics} from './analysis-join-policy';
import {
  getResidentSpatialJoinPointTransform,
  normalizeResidentSpatialJoinCrs,
  type ResidentSpatialJoinCrs,
  type ResidentSpatialJoinPointTransform
} from './resident-spatial-join-reproject';
import type {
  ResidentJoinColumnFormat,
  ResidentJoinField,
  ResidentJoinSourceMetadata
} from './resident-join-plan';

/**
 * Structural limits only; row counts, polygon counts and pair counts have no cap. Data-dependent
 * memory (join working set, result columns, retained Arrow copy) is admitted by
 * `reserveMemory` at execution time and fails with SQL guidance when it does not fit.
 */
export const MAXIMUM_RESIDENT_SPATIAL_JOIN_COLUMNS = 32;
export const MAXIMUM_RESIDENT_SPATIAL_JOIN_AGGREGATIONS = 16;

export type ResidentSpatialJoinPredicate = 'intersects' | 'within';
export type ResidentSpatialJoinGpuTechnique = 'count' | 'sum' | 'average' | 'maximum' | 'minimum';
/** `merge` unions the matched polygon rows on the CPU from accounted pair readback. */
export type ResidentSpatialJoinTechnique = ResidentSpatialJoinGpuTechnique | 'merge';
/** Results with merge are read back, so they stay within the readback row cap. */
export const MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_ROW_COUNT = 262144;
/** Matched (point, polygon) pairs that one merge may read back (8 bytes each). */
export const MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_PAIR_COUNT = 4194304;
export type ResidentSpatialJoinGeo = {
  kind: string;
  latField?: string;
  lngField?: string;
  fieldName?: string;
} | null;
export type ResidentSpatialJoinOperation = {
  id: string;
  implementation?: string;
  leftDataId: string;
  rightDataId: string;
  leftGeo: ResidentSpatialJoinGeo;
  rightGeo: ResidentSpatialJoinGeo;
  predicate?: string;
  /** Declared coordinate systems; both default to EPSG:4326 and must match (no reprojection). */
  leftCrs?: string;
  rightCrs?: string;
  aggregations: Readonly<Record<string, string>>;
  leftColumns?: readonly string[];
  resultId: string;
  resultLabel: string;
};
export type ResidentSpatialJoinSourceMetadata = ResidentJoinSourceMetadata & {
  /** Resident point coordinate roles, present only for a point publication. */
  coordinates?: {longitude: string; latitude: string};
  /** Resident geometry column, present only for a geometry publication. */
  geometryColumn?: string;
};
export type ResidentSpatialJoinColumn =
  | {
      kind: 'left';
      sourceColumn: string;
      outputColumn: string;
      type: string;
      analyzerType: string;
      format: ResidentJoinColumnFormat;
      dictionary: boolean;
    }
  | {kind: 'count'; outputColumn: string; type: 'integer'; analyzerType: 'integer'}
  | {
      kind: 'aggregate';
      sourceColumn: string;
      technique: ResidentSpatialJoinTechnique;
      outputColumn: string;
      type: string;
      analyzerType: string;
      /** Source storage; count needs none. */
      sourceFormat?: ResidentJoinColumnFormat;
    };
export type ResidentSpatialJoinProvenance = {
  schemaVersion: 1;
  kind: 'spatial-join';
  left: {datasetId: string; revision: number};
  right: {datasetId: string; revision: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  predicate: ResidentSpatialJoinPredicate;
  leftGeo: {latField: string; lngField: string};
  rightGeo: {fieldName: string};
  columns: readonly {
    kind: 'left' | 'count' | 'aggregate';
    column: string | null;
    technique: ResidentSpatialJoinTechnique | null;
    name: string;
  }[];
  fields: readonly ResidentJoinField[];
  numericalPolicy: 'float32';
  /** Declared coordinate systems; points are transformed into the polygon CRS when they differ. */
  crs?: {
    points: ResidentSpatialJoinCrs;
    polygons: ResidentSpatialJoinCrs;
    pointTransform: ResidentSpatialJoinPointTransform;
  };
  boundaryPolicy: 'closed' | 'open';
  rowOrder: 'left-source-order';
};
export type ResidentSpatialJoinPlan = {
  left: {datasetId: string; revision: number; rowCount: number};
  right: {datasetId: string; revision: number; rowCount: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  predicate: ResidentSpatialJoinPredicate;
  columns: readonly ResidentSpatialJoinColumn[];
  provenance: ResidentSpatialJoinProvenance;
  derivedDataset: {type: 'spatialJoin'; sourceDataIds: [string, string]; operationId: string};
};

const SQL_GUIDANCE =
  ' Use SQL for a complementary query with unsupported geometry, predicates, aggregations or sizes.';
const TECHNIQUES = ['count', 'sum', 'average', 'maximum', 'minimum'] as const;

/**
 * Native semantics (kepler.gl table/dataset-ops/spatial-join.js): one result row per left row, in
 * left order; left geometry fields are always included; `count` counts matched right rows; each
 * aggregation reduces the matched right values (count includes nulls, others skip them and give
 * null when nothing is defined). This slice admits resident longitude/latitude points on the left
 * and resident polygon rows on the right, with the intersects and within predicates.
 */
export function createResidentSpatialJoinPlan(
  operation: ResidentSpatialJoinOperation,
  left: ResidentSpatialJoinSourceMetadata,
  right: ResidentSpatialJoinSourceMetadata
): ResidentSpatialJoinPlan {
  if (left.datasetId !== operation.leftDataId || right.datasetId !== operation.rightDataId)
    reject('The Spatial Join sources have changed.');
  if (left.datasetId === right.datasetId) reject('Spatial Join requires two different datasets.');
  const predicate = operation.predicate ?? 'intersects';
  if (predicate !== 'intersects' && predicate !== 'within')
    reject(`Spatial Join predicate ${predicate} is unavailable on the GPU.`);
  // Points are moved into the polygon CRS, so the boundary policy is planned in that single CRS.
  // Same-CRS EPSG:4326 and EPSG:3857 joins run directly; mixed declarations reproject the points
  // on the GPU (4326, 3857, UTM and general transverse Mercator). Any other CRS is refused rather than guessed.
  const pointCrs = normalizeResidentSpatialJoinCrs(operation.leftCrs ?? 'EPSG:4326');
  const polygonCrs = normalizeResidentSpatialJoinCrs(operation.rightCrs ?? 'EPSG:4326');
  if (!pointCrs || !polygonCrs)
    reject(
      `Spatial Join runs on EPSG:4326, EPSG:3857, UTM EPSG codes or proj4 longlat/utm/tmerc parameters on a WGS84 or GRS80 ellipsoid only (declared ${operation.leftCrs ?? 'EPSG:4326'}, ${operation.rightCrs ?? 'EPSG:4326'}).`
    );
  const {semantics, refusal} = planSpatialJoinSemantics(predicate, polygonCrs, polygonCrs);
  if (!semantics) reject(refusal ?? 'Spatial Join semantics are unavailable.');
  const pointTransform = getResidentSpatialJoinPointTransform(pointCrs, polygonCrs);
  const operationId = readText(operation.id, 'operation ID');
  const resultId = readText(operation.resultId, 'result ID');
  const resultLabel = readText(operation.resultLabel, 'result name');
  if (resultId === left.datasetId || resultId === right.datasetId)
    reject('Spatial Join must create a separate result dataset.');
  const leftGeo = operation.leftGeo;
  const rightGeo = operation.rightGeo;
  if (
    leftGeo?.kind !== 'latlng' ||
    !leftGeo.latField ||
    !leftGeo.lngField ||
    left.coordinates?.latitude !== leftGeo.latField ||
    left.coordinates.longitude !== leftGeo.lngField
  )
    reject(
      'Resident Spatial Join requires the left geometry to be the published longitude/latitude point roles.'
    );
  if (
    rightGeo?.kind !== 'geojson' ||
    !rightGeo.fieldName ||
    right.geometryColumn !== rightGeo.fieldName ||
    right.fields.find(field => field.name === rightGeo.fieldName)?.type !== 'geojson'
  )
    reject('Resident Spatial Join requires the right geometry to be a published polygon column.');
  // Counts must be positive safe integers (an empty side has no native result to mirror).
  if (!Number.isSafeInteger(left.rowCount) || left.rowCount < 1)
    reject('Spatial Join left input must have at least 1 row.');
  if (!Number.isSafeInteger(right.rowCount) || right.rowCount < 1)
    reject('Spatial Join polygons must have at least 1 row.');

  const geoNames = new Set([leftGeo.latField, leftGeo.lngField]);
  const includedLeft = left.fields.filter(
    field =>
      operation.leftColumns === undefined ||
      geoNames.has(field.name) ||
      operation.leftColumns.includes(field.name)
  );
  const used = new Set<string>();
  const columns: ResidentSpatialJoinColumn[] = [];
  for (const field of includedLeft) {
    const format = left.columns[field.name]?.format;
    const dictionary = Object.hasOwn(left.dictionaries, field.name)
      ? left.dictionaries[field.name]
      : undefined;
    const supported =
      (format === 'float32' && !dictionary && ['integer', 'real'].includes(field.type)) ||
      (format === 'sint32' && !dictionary && field.type === 'integer') ||
      (format === 'uint32' &&
        (dictionary ? field.type === 'string' && !dictionary.ordered : field.type === 'integer'));
    if (!supported)
      reject(`Resident Spatial Join cannot carry left column ${field.name} (${field.type}).`);
    columns.push({
      kind: 'left',
      sourceColumn: field.name,
      outputColumn: uniqueName(field.name, used),
      type: field.type,
      analyzerType: field.analyzerType || field.type,
      format: format as ResidentJoinColumnFormat,
      dictionary: Boolean(dictionary)
    });
  }
  columns.push({
    kind: 'count',
    outputColumn: uniqueName('count', used),
    type: 'integer',
    analyzerType: 'integer'
  });
  const entries = Object.entries(operation.aggregations);
  if (entries.length > MAXIMUM_RESIDENT_SPATIAL_JOIN_AGGREGATIONS)
    reject(
      `Spatial Join supports at most ${MAXIMUM_RESIDENT_SPATIAL_JOIN_AGGREGATIONS} aggregations.`
    );
  for (const [name, technique] of entries) {
    const field = right.fields.find(candidate => candidate.name === name);
    if (!field) reject(`Spatial Join aggregation field ${name} is unavailable.`);
    if (technique === 'merge') {
      // Native merge collects the matched geometries; only the published polygon column has them.
      if (name !== right.geometryColumn || field.type !== 'geojson')
        reject(`Spatial Join merge requires the published polygon column: ${name}.`);
      if (left.rowCount > MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_ROW_COUNT)
        reject(
          `Spatial Join merge reads the result back and supports at most ${MAXIMUM_RESIDENT_SPATIAL_JOIN_MERGE_ROW_COUNT} left rows; uncheck merge for larger inputs.`
        );
      columns.push({
        kind: 'aggregate',
        sourceColumn: name,
        technique: 'merge',
        outputColumn: uniqueName(`${name}_merge`, used),
        type: 'geojson',
        analyzerType: field.analyzerType || field.type
      });
      continue;
    }
    if (!(TECHNIQUES as readonly string[]).includes(technique))
      reject(`Spatial Join aggregation ${technique} is unavailable on the GPU.`);
    const format = right.columns[name]?.format as ResidentJoinColumnFormat | undefined;
    const numeric =
      !Object.hasOwn(right.dictionaries, name) &&
      (format === 'float32' || format === 'sint32' || format === 'uint32') &&
      ['integer', 'real'].includes(field.type);
    if (technique !== 'count' && !numeric)
      reject(`Spatial Join ${technique} requires a prepared numeric field: ${name}.`);
    const kind = technique as ResidentSpatialJoinGpuTechnique;
    columns.push({
      kind: 'aggregate',
      sourceColumn: name,
      technique: kind,
      outputColumn: uniqueName(`${name}_${technique}`, used),
      // Native resultFieldType: count integer, average real, otherwise the source type.
      type: kind === 'count' ? 'integer' : kind === 'average' ? 'real' : field.type,
      analyzerType: field.analyzerType || field.type,
      ...(numeric ? {sourceFormat: format} : {})
    });
  }
  if (columns.length > MAXIMUM_RESIDENT_SPATIAL_JOIN_COLUMNS)
    reject(
      `Spatial Join supports at most ${MAXIMUM_RESIDENT_SPATIAL_JOIN_COLUMNS} output columns.`
    );
  const fields = columns.map(column => ({
    name: column.outputColumn,
    type: column.type,
    analyzerType: column.analyzerType
  }));
  const provenance: ResidentSpatialJoinProvenance = {
    schemaVersion: 1,
    kind: 'spatial-join',
    left: {datasetId: left.datasetId, revision: left.revision},
    right: {datasetId: right.datasetId, revision: right.revision},
    operationId,
    resultId,
    resultLabel,
    predicate,
    leftGeo: {latField: leftGeo.latField, lngField: leftGeo.lngField},
    rightGeo: {fieldName: rightGeo.fieldName},
    columns: columns.map(column => ({
      kind: column.kind,
      column: column.kind === 'count' ? null : column.sourceColumn,
      technique: column.kind === 'aggregate' ? column.technique : null,
      name: column.outputColumn
    })),
    fields,
    numericalPolicy: 'float32',
    crs: {points: pointCrs, polygons: polygonCrs, pointTransform},
    boundaryPolicy: semantics.boundary === 'inclusive' ? 'closed' : 'open',
    rowOrder: 'left-source-order'
  };
  return {
    left: {datasetId: left.datasetId, revision: left.revision, rowCount: left.rowCount},
    right: {datasetId: right.datasetId, revision: right.revision, rowCount: right.rowCount},
    operationId,
    resultId,
    resultLabel,
    predicate,
    columns,
    provenance,
    derivedDataset: {
      type: 'spatialJoin',
      sourceDataIds: [left.datasetId, right.datasetId],
      operationId
    }
  };
}

/**
 * Native aggregateValues over matched right values, in matched order. Count includes nulls; the
 * rest skip nulls and give null when nothing is defined. Sums use double accumulation and round
 * once to the Float32 output policy.
 */
export function aggregateResidentSpatialJoinValues(
  values: readonly (number | null)[],
  technique: ResidentSpatialJoinGpuTechnique
): number | null {
  if (technique === 'count') return values.length;
  let defined = 0;
  let sum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    if (value === null) continue;
    defined++;
    sum += value;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  if (!defined) return null;
  return technique === 'sum'
    ? sum
    : technique === 'average'
      ? sum / defined
      : technique === 'maximum'
        ? maximum
        : minimum;
}

function uniqueName(base: string, used: Set<string>): string {
  let name = base;
  let suffix = 1;
  while (used.has(name)) name = `${base}_${suffix++}`;
  used.add(name);
  return name;
}
function readText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 4096)
    reject(`Invalid Spatial Join ${label}.`);
  return value;
}
function reject(message: string): never {
  throw new Error(`${message}${SQL_GUIDANCE}`);
}
