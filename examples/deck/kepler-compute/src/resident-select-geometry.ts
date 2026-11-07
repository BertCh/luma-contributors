// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {Feature, Geometry} from 'geojson';
import type {KeplerDatasetRegistry, KeplerSourceRow} from '@luma-contributors/kepler';
import {
  getGeoArrowRowViewFromArrowVector,
  makeArrowVectorFromGeoArrowColumn
} from '@luma-contributors/graph/arrow';
import type {GeoArrowEncoding} from '@math.gl/geospatial/geoarrow';
import {parseWKB, visitWKB} from '@math.gl/geospatial';
import {cellToBoundary} from 'h3-js';
import {prepareH3Source, type H3SourceAccessor} from './h3-source';

export type ResidentSelectedGeometryRequest = {
  registry: KeplerDatasetRegistry;
  datasetId: string;
  revision: number;
  sourceRowIndex: number;
  kind: 'geojson' | 'hexagonId';
  column: string;
  /** Caller creates a fresh editor ID for each selection, as pinned MapPopover does. */
  featureId: string;
  maximumPayloadBytes: number;
  maximumVertices?: number;
  h3Accessor?: H3SourceAccessor;
  /** Explicit encoding for already decoded sources; otherwise use the retained Arrow field. */
  geometryEncoding?: GeoArrowEncoding;
};
export type ResidentSelectedGeometryResult =
  | {
      status: 'ready';
      feature: Feature<Geometry>;
      payloadByteLength: number;
      source: KeplerSourceRow;
      /** Recheck immediately before dispatching the native editor actions. No lease is retained. */
      isCurrent: () => boolean;
    }
  | {status: 'stale' | 'unsupported' | 'budget-exceeded'; reason: string};

class GeometryPayloadLimit extends Error {}
class GeometryUnavailable extends Error {}
const GEOMETRY_TYPES: Record<string, Geometry['type']> = {
  'geoarrow.point': 'Point',
  'geoarrow.multipoint': 'MultiPoint',
  'geoarrow.linestring': 'LineString',
  'geoarrow.multilinestring': 'MultiLineString',
  'geoarrow.polygon': 'Polygon',
  'geoarrow.multipolygon': 'MultiPolygon'
};

/**
 * One clicked retained-source row only. Never calls DataRow, rowAsArray, or whole-column conversion.
 * The payload budget bounds physical selected-cell preparation before decoding, then the exact
 * detached JSON payload. Native coordinate lists are conservatively charged 24 bytes per scalar.
 * Schema/batch metadata may be traversed; unrelated row values and dictionary entries are not read.
 * A ready result owns its detached feature, and callers recheck isCurrent() before editor dispatch.
 */
export async function readResidentSelectedGeometry(
  request: ResidentSelectedGeometryRequest
): Promise<ResidentSelectedGeometryResult> {
  if (
    !Number.isSafeInteger(request.maximumPayloadBytes) ||
    request.maximumPayloadBytes < 1 ||
    !Number.isSafeInteger(request.maximumVertices ?? 10000) ||
    (request.maximumVertices ?? 10000) < 1 ||
    request.featureId.length > request.maximumPayloadBytes ||
    !request.featureId.trim()
  )
    return {
      status: 'unsupported',
      reason: 'Selection requires a fresh feature ID and explicit positive payload limits.'
    };
  const snapshot = request.registry.getDataset(request.datasetId);
  if (!snapshot || snapshot.revision !== request.revision)
    return {status: 'stale', reason: 'The clicked dataset revision is no longer current.'};
  const lease = request.registry.acquireDataset(request.datasetId);
  const isCurrent = () =>
    request.registry.getDataset(request.datasetId) === snapshot &&
    request.registry.deviceEpoch === snapshot.deviceEpoch;
  try {
    if (!lease.isValid() || !isCurrent())
      return {status: 'stale', reason: 'The clicked source expired.'};
    if (
      snapshot.rowIdentity !== 'source-order' ||
      !Number.isSafeInteger(request.sourceRowIndex) ||
      request.sourceRowIndex < 0 ||
      request.sourceRowIndex >= snapshot.rowCount
    )
      throw new GeometryUnavailable('Selection requires a valid original source-row identity.');
    const table = snapshot.source?.kind === 'arrow' ? snapshot.source.value : undefined;
    if (!(table instanceof arrow.Table))
      throw new GeometryUnavailable('This dataset has no retained Arrow source.');
    const column = table.getChild(request.column);
    const field = table.schema.fields.find(candidate => candidate.name === request.column);
    if (!column || !field)
      throw new GeometryUnavailable('The selected geometry column is not retained.');
    if (table.numRows !== snapshot.rowCount)
      throw new GeometryUnavailable('Retained source row identities do not match the dataset.');
    const row = column.slice(request.sourceRowIndex, request.sourceRowIndex + 1);
    if (!row.isValid(0)) throw new GeometryUnavailable('The clicked source geometry is null.');
    const budget = new PayloadBudget(request.maximumPayloadBytes, request.maximumVertices ?? 10000);
    // Check physical offsets/counts before UTF8 decode, JSON.parse or Arrow list materialization.
    for (const data of row.data) inspectArrowStorage(data, budget, 0);
    const metadataText = field.metadata.get('ARROW:extension:metadata');
    if (metadataText) {
      budget.string(metadataText);
      const metadata: unknown = JSON.parse(metadataText);
      const crs =
        metadata && typeof metadata === 'object' && 'crs' in metadata ? metadata.crs : undefined;
      if (crs !== undefined && crs !== null && crs !== 'EPSG:4326' && crs !== 'OGC:CRS84')
        throw new GeometryUnavailable(
          'Select Geometry requires geographic longitude/latitude source coordinates.'
        );
    }
    let feature: Feature<Geometry>;
    if (request.kind === 'hexagonId') {
      const prepared = prepareH3Source([{id: 'selected', values: [row.get(0)]}], {
        accessor: request.h3Accessor
      });
      const cellId = prepared.batches[0].canonicalIds[0];
      if (!cellId)
        throw new GeometryUnavailable(
          prepared.invalidRows[0]?.reason ?? 'The clicked H3 source row is empty.'
        );
      // Pinned idToPolygonGeo({id}, {isClosed:true}); no coverage, wrapping or viewport footprint.
      feature = {
        type: 'Feature',
        geometry: {type: 'Polygon', coordinates: [cellToBoundary(cellId, true)]},
        properties: {isClosed: true}
      };
    } else {
      const encoding = request.geometryEncoding ?? field.metadata.get('ARROW:extension:name');
      if (encoding === 'geoarrow.wkb' || (!encoding && arrow.DataType.isBinary(row.type))) {
        const value: unknown = row.get(0);
        if (!(value instanceof Uint8Array))
          throw new GeometryUnavailable('The clicked WKB geometry is empty.');
        visitWKB(
          value,
          {
            coordinate: () => budget.vertices(1),
            geometry: header => {
              if (header.dimension !== 'xy' && header.dimension !== 'xyz')
                throw new GeometryUnavailable(
                  'WKB measure coordinates require an explicit geometry adapter.'
                );
              if (header.srid !== undefined && header.srid !== 0 && header.srid !== 4326)
                throw new GeometryUnavailable(
                  'Select Geometry requires geographic WKB coordinates.'
                );
            }
          },
          {maximumDepth: 32}
        );
        feature = {
          type: 'Feature',
          geometry: parseWKB(value, {maximumDepth: 32}).geometry as Geometry,
          properties: null
        };
      } else if (encoding && (GEOMETRY_TYPES[encoding] || encoding === 'geoarrow.geometry')) {
        const view = getGeoArrowRowViewFromArrowVector(row, 0, {
          encoding: encoding as GeoArrowEncoding
        });
        const geometryType = GEOMETRY_TYPES[view.column.encoding];
        if (!geometryType || (view.column.dimension !== 'xy' && view.column.dimension !== 'xyz'))
          throw new GeometryUnavailable(
            'This GeoArrow geometry type or coordinate dimension has no selection adapter.'
          );
        const value: unknown = makeArrowVectorFromGeoArrowColumn(view.column).get(0);
        if (value == null) throw new GeometryUnavailable('The clicked geometry is empty.');
        feature = {
          type: 'Feature',
          geometry: {type: geometryType, coordinates: copyCoordinates(value)} as Geometry,
          properties: null
        };
      } else if (encoding) {
        throw new GeometryUnavailable(`Select Geometry does not support ${encoding}.`);
      } else {
        const value: unknown = row.get(0);
        const parsed: unknown =
          typeof value === 'string' ? JSON.parse(value) : copyArrowValue(value);
        feature = normalizeFeature(parsed);
      }
    }
    let vertexCount = 0;
    validateGeometry(
      feature.geometry,
      () => {
        if (++vertexCount > (request.maximumVertices ?? 10000))
          throw new GeometryPayloadLimit('Selected geometry exceeds the vertex budget.');
      },
      0
    );
    // Clone only the admitted row; editor mutations must not mutate retained source properties.
    const selected = {...feature, id: request.featureId};
    inspectJSONPayload(selected, new PayloadBudget(request.maximumPayloadBytes), 0);
    const serialized = JSON.stringify(selected);
    const payloadByteLength = new TextEncoder().encode(serialized).byteLength;
    if (payloadByteLength > request.maximumPayloadBytes)
      throw new GeometryPayloadLimit('Selected geometry exceeds the payload byte budget.');
    const detached: Feature<Geometry> = JSON.parse(serialized);
    let sourceRowOffset = 0;
    let batchId: string | undefined;
    for (const batch of snapshot.batches) {
      const offset = batch.sourceRowOffset ?? sourceRowOffset;
      if (request.sourceRowIndex >= offset && request.sourceRowIndex < offset + batch.rowCount) {
        batchId = batch.id;
        break;
      }
      sourceRowOffset += batch.rowCount;
    }
    if (batchId === undefined)
      throw new GeometryUnavailable('The clicked source batch is unavailable.');
    await Promise.resolve();
    if (!lease.isValid() || !isCurrent())
      return {status: 'stale', reason: 'The dataset changed while preparing selected geometry.'};
    return {
      status: 'ready',
      feature: detached,
      payloadByteLength,
      source: {
        datasetId: snapshot.id,
        revision: snapshot.revision,
        batchId,
        sourceRowIndex: request.sourceRowIndex
      },
      isCurrent
    };
  } catch (error) {
    return {
      status: error instanceof GeometryPayloadLimit ? 'budget-exceeded' : 'unsupported',
      reason: error instanceof Error ? error.message : String(error)
    };
  } finally {
    await lease.release();
  }
}

class PayloadBudget {
  private consumed = 0;
  private vertexCount = 0;
  constructor(
    readonly maximum: number,
    readonly maximumVertices = 10000
  ) {}
  vertices(count: number): void {
    if (
      !Number.isSafeInteger(count) ||
      count < 0 ||
      count > this.maximumVertices - this.vertexCount
    )
      throw new GeometryPayloadLimit(
        'Selected source geometry exceeds the vertex preparation budget.'
      );
    this.vertexCount += count;
  }
  add(bytes: number): void {
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > this.maximum - this.consumed)
      throw new GeometryPayloadLimit(
        'Selected source cell exceeds the payload preparation budget.'
      );
    this.consumed += bytes;
  }
  string(value: string): void {
    if (value.length > this.maximum)
      throw new GeometryPayloadLimit('Selected source text exceeds the payload budget.');
    this.add(new TextEncoder().encode(value).byteLength);
  }
}

/** Bound the detached editor payload before JSON.stringify allocates its output. */
function inspectJSONPayload(value: unknown, budget: PayloadBudget, depth: number): void {
  if (depth > 32)
    throw new GeometryPayloadLimit('Selected feature properties exceed the nesting budget.');
  if (typeof value === 'string') {
    budget.add(2);
    for (let index = 0; index < value.length; index++) {
      const code = value.charCodeAt(index);
      if (
        code === 34 ||
        code === 92 ||
        code === 8 ||
        code === 9 ||
        code === 10 ||
        code === 12 ||
        code === 13
      )
        budget.add(2);
      else if (code < 32) budget.add(6);
      else if (code < 128) budget.add(1);
      else if (code < 2048) budget.add(2);
      else if (
        code >= 0xd800 &&
        code <= 0xdbff &&
        index + 1 < value.length &&
        value.charCodeAt(index + 1) >= 0xdc00 &&
        value.charCodeAt(index + 1) <= 0xdfff
      ) {
        budget.add(4);
        index++;
      } else if (code >= 0xd800 && code <= 0xdfff) budget.add(6);
      else budget.add(3);
    }
  } else if (typeof value === 'number' || typeof value === 'boolean' || value === null) {
    budget.add(JSON.stringify(value).length);
  } else if (Array.isArray(value)) {
    budget.add(2 + Math.max(0, value.length - 1));
    for (const item of value) inspectJSONPayload(item ?? null, budget, depth + 1);
  } else if (value && typeof value === 'object') {
    budget.add(2);
    let count = 0;
    for (const key in value) {
      if (!Object.hasOwn(value, key)) continue;
      const item: unknown = Reflect.get(value, key);
      if (item === undefined) continue;
      budget.add(1 + Number(count++ > 0));
      inspectJSONPayload(key, budget, depth + 1);
      inspectJSONPayload(item, budget, depth + 1);
    }
  } else throw new GeometryUnavailable('Selected feature properties are not JSON serializable.');
}

/** Physical Arrow views use sliced offsets/values; nested List children remain shared. */
function inspectArrowStorage(data: arrow.Data, budget: PayloadBudget, depth: number): void {
  if (depth > 32) throw new GeometryPayloadLimit('Selected geometry exceeds the nesting budget.');
  budget.add(data.length * 2);
  const type = data.type;
  if (
    arrow.DataType.isUtf8(type) ||
    arrow.DataType.isLargeUtf8(type) ||
    arrow.DataType.isBinary(type) ||
    arrow.DataType.isLargeBinary(type)
  ) {
    budget.add(Number(data.valueOffsets[data.length]) - Number(data.valueOffsets[0]));
  } else if (arrow.DataType.isList(type)) {
    const start = Number(data.valueOffsets[0]);
    const end = Number(data.valueOffsets[data.length]);
    inspectArrowStorage(data.children[0].slice(start, end - start), budget, depth + 1);
  } else if (arrow.DataType.isFixedSizeList(type)) {
    if (arrow.DataType.isFloat(data.children[0].type)) budget.vertices(data.length);
    inspectArrowStorage(data.children[0].slice(0, data.length * type.listSize), budget, depth + 1);
  } else if (arrow.DataType.isStruct(type)) {
    if (
      type.children.some(field => field.name === 'x') &&
      type.children.some(field => field.name === 'y')
    )
      budget.vertices(data.length);
    for (const child of data.children)
      inspectArrowStorage(child.slice(0, data.length), budget, depth + 1);
  } else if (arrow.DataType.isDenseUnion(type)) {
    for (let index = 0; index < data.length; index++) {
      const childIndex = type.typeIdToChildIndex[data.typeIds[index]];
      const child = data.children[childIndex];
      if (!child) throw new GeometryUnavailable('Invalid GeoArrow union child.');
      inspectArrowStorage(child.slice(data.valueOffsets[index], 1), budget, depth + 1);
    }
  } else if (arrow.DataType.isDictionary(type)) {
    for (let index = 0; index < data.length; index++) {
      const dictionaryRow = Number(data.values[index]);
      const dictionary = data.dictionary?.slice(dictionaryRow, dictionaryRow + 1);
      if (!dictionary || dictionary.length !== 1)
        throw new GeometryUnavailable('Invalid source dictionary row.');
      for (const child of dictionary.data) inspectArrowStorage(child, budget, depth + 1);
    }
  } else if (
    arrow.DataType.isFloat(type) ||
    arrow.DataType.isInt(type) ||
    arrow.DataType.isBool(type) ||
    arrow.DataType.isNull(type)
  ) {
    budget.add(data.length * 24);
  } else {
    throw new GeometryUnavailable(
      `Select Geometry cannot safely preflight Arrow ${type.toString()}.`
    );
  }
}
function copyArrowValue(value: unknown): unknown {
  if (value instanceof arrow.Vector) return Array.from(value, copyArrowValue);
  if (value instanceof arrow.StructRow)
    return Object.fromEntries([...value].map(([key, item]) => [key, copyArrowValue(item)]));
  return value;
}
function copyCoordinates(value: unknown): unknown {
  if (value instanceof arrow.Vector || Array.isArray(value))
    return Array.from(value, copyCoordinates);
  if (value && typeof value === 'object' && 'x' in value && 'y' in value)
    return 'z' in value ? [value.x, value.y, value.z] : [value.x, value.y];
  return value;
}
function normalizeFeature(value: unknown): Feature<Geometry> {
  if (!value || typeof value !== 'object' || !('type' in value))
    throw new GeometryUnavailable('The selected cell is not GeoJSON.');
  if (value.type === 'FeatureCollection') {
    if (!('features' in value) || !Array.isArray(value.features) || !value.features.length)
      throw new GeometryUnavailable('The selected FeatureCollection is empty.');
    return normalizeFeature(value.features[0]);
  }
  if (value.type === 'Feature') return {properties: null, ...value} as Feature<Geometry>;
  if (
    [
      'Point',
      'MultiPoint',
      'LineString',
      'MultiLineString',
      'Polygon',
      'MultiPolygon',
      'GeometryCollection'
    ].includes(String(value.type))
  )
    return {type: 'Feature', geometry: value as Geometry, properties: {}};
  throw new GeometryUnavailable('This source cell has no supported GeoJSON geometry.');
}
function validateGeometry(geometry: Geometry, visitVertex: () => void, depth: number): void {
  if (!geometry || depth > 32)
    throw new GeometryUnavailable('The selected geometry is empty or too deeply nested.');
  if (geometry.type === 'GeometryCollection') {
    for (const child of geometry.geometries) validateGeometry(child, visitVertex, depth + 1);
    return;
  }
  const levels = {
    Point: 0,
    MultiPoint: 1,
    LineString: 1,
    MultiLineString: 2,
    Polygon: 2,
    MultiPolygon: 3
  }[geometry.type];
  if (levels === undefined) throw new GeometryUnavailable('Unsupported selected geometry type.');
  const visit = (value: unknown, remaining: number): void => {
    if (!Array.isArray(value))
      throw new GeometryUnavailable('Geometry coordinates must be arrays.');
    if (remaining) {
      for (const child of value) visit(child, remaining - 1);
      return;
    }
    if (
      value.length < 2 ||
      value.length > 3 ||
      value.some(item => typeof item !== 'number' || !Number.isFinite(item))
    )
      throw new GeometryUnavailable('Selected geometry requires finite XY or XYZ coordinates.');
    visitVertex();
  };
  visit(geometry.coordinates, levels);
}
