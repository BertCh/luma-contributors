// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataType, Vector} from 'apache-arrow';
import {
  iterateGeoArrowBatchesFromArrowVector,
  type MakeGeoArrowColumnFromArrowVectorOptions
} from '@luma.gl/graph/arrow';
import type {GPUSpatialFrame} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {CPUSpatialJoinGeometry} from './cpu-feature-join';

type NativeBatch =
  ReturnType<typeof iterateGeoArrowBatchesFromArrowVector> extends IterableIterator<infer Batch>
    ? Batch
    : never;
type NativeArray = NativeBatch['column']['chunks'][number];

/** Native XY GeoArrow adaptation; sourceIds are aligned to the supplied vector, including nulls. */
export type PrepareGeometryInputOptions = MakeGeoArrowColumnFromArrowVectorOptions & {
  id?: string;
  sourceRowOffset?: number;
  sourceIds?: Uint32Array;
  frame?: GPUSpatialFrame;
  /** Default reject-loss. allow-loss explicitly permits float64 rounding and coordinate collapse. */
  precision?: 'reject-loss' | 'allow-loss';
};

/** Metric XY coordinates, compact hierarchy, original identity and explicit precision diagnostics. */
export type PreparedGeometryInput = {
  geometry: CPUSpatialJoinGeometry;
  sourceRows: Uint32Array;
  sourceIds: Uint32Array;
  /** 0 for null; an empty but nonnull feature has validity 1 and empty 1. */
  validity: Uint32Array;
  empty: Uint32Array;
  batchRowOffsets: Uint32Array;
  frame?: GPUSpatialFrame;
  roundedCoordinateCount: number;
  /** Distinct original XY positions beyond one per rounded XY; repeated vertices do not add loss. */
  collapsedPositionCount: number;
};

/** Packs supported native GeoArrow Point/MultiPoint, LineString/MultiLineString and Polygon/MultiPolygon columns. */
export function prepareGeometryInput<T extends DataType>(
  column: Vector<T>,
  options: PrepareGeometryInputOptions = {}
): PreparedGeometryInput {
  const id = options.id ?? 'prepareGeometryInput';
  const fieldEncoding = options.field?.metadata.get('ARROW:extension:name');
  validateRequirement(
    id,
    !fieldEncoding || options.encoding === undefined || options.encoding === fieldEncoding,
    `encoding conflicts with field encoding ${fieldEncoding}; received ${options.encoding}`
  );
  const serializedMetadata = options.field?.metadata.get('ARROW:extension:metadata');
  const fieldMetadata: unknown = serializedMetadata ? JSON.parse(serializedMetadata) : {};
  validateRequirement(
    id,
    Boolean(fieldMetadata) && typeof fieldMetadata === 'object' && !Array.isArray(fieldMetadata),
    'field metadata must be an object'
  );
  const metadata = fieldMetadata as Record<string, unknown>;
  const canonicalize = (value: unknown): string => {
    if (Array.isArray(value)) return `[${value.map(canonicalize).join(',')}]`;
    if (value && typeof value === 'object')
      return `{${Object.entries(value)
        .sort(([first], [second]) => first.localeCompare(second))
        .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalize(entry)}`)
        .join(',')}}`;
    return JSON.stringify(value) ?? 'undefined';
  };
  for (const [name, value] of Object.entries(options.metadata ?? {}))
    validateRequirement(
      id,
      !(name in metadata) || canonicalize(value) === canonicalize(metadata[name]),
      `metadata.${name} conflicts with field declaration; received ${canonicalize(value)}`
    );
  const adapterOptions = {...options, metadata: {...metadata, ...options.metadata}};
  const rowOffset = options.sourceRowOffset ?? 0;
  validateCapacity(id, 'sourceRowOffset', rowOffset, 0, 0xffffffff - column.length);
  validateRequirement(
    id,
    !options.sourceIds || options.sourceIds.length === column.length,
    `sourceIds length must equal column length ${column.length}; received ${options.sourceIds?.length}`
  );
  const precision = options.precision ?? 'reject-loss';
  validateRequirement(
    id,
    ['reject-loss', 'allow-loss'].includes(precision),
    `precision must be reject-loss or allow-loss; received ${precision}`
  );
  const positions: number[] = [],
    featureOffsets = [0],
    lineOffsets = [0],
    polygonOffsets = [0],
    ringOffsets = [0];
  const validity = new Uint32Array(column.length),
    empty = new Uint32Array(column.length),
    batchRowOffsets: number[] = [];
  let kind: CPUSpatialJoinGeometry['kind'] | undefined;
  let roundedCoordinateCount = 0,
    collapsedPositionCount = 0;
  const represented = new Map<string, Set<string>>();
  let singlePoints = true;
  const valid = (array: NativeArray, row: number) => {
    if (!array.validity) return true;
    const bit = (array.validity.bitOffset ?? 0) + row;
    return Boolean(array.validity.values[bit >> 3] & (1 << (bit & 7)));
  };
  const scalar = (array: NativeArray, row: number): number => {
    validateRequirement(
      id,
      array.kind === 'primitive' && row >= 0 && row < array.length && valid(array, row),
      'coordinates require valid primitive XY values'
    );
    if (array.kind !== 'primitive') return Number.NaN;
    return Number(array.values[(array.offset ?? 0) + row * (array.stride ?? 1)]);
  };
  const appendPosition = (array: NativeArray, row: number, allowEmpty = false) => {
    validateRequirement(
      id,
      row >= 0 && row < array.length && valid(array, row),
      'coordinate row must be valid and in range'
    );
    let x: number, y: number;
    if (array.kind === 'fixed-size-list') {
      validateRequirement(
        id,
        array.size === 2,
        `interleaved coordinate size must be 2; received ${array.size}`
      );
      x = scalar(array.child, ((array.offset ?? 0) + row) * 2);
      y = scalar(array.child, ((array.offset ?? 0) + row) * 2 + 1);
    } else {
      validateRequirement(
        id,
        array.kind === 'struct',
        'coordinates must use interleaved fixed-size-list or separated struct storage'
      );
      if (array.kind !== 'struct') return;
      x = scalar(array.children['x'], (array.offset ?? 0) + row);
      y = scalar(array.children['y'], (array.offset ?? 0) + row);
    }
    if (allowEmpty && Number.isNaN(x) && Number.isNaN(y)) return;
    validateRequirement(
      id,
      Number.isFinite(x) &&
        Number.isFinite(y) &&
        Number.isFinite(Math.fround(x)) &&
        Number.isFinite(Math.fround(y)),
      `positions must remain finite float32 XY; received ${x}, ${y}`
    );
    const roundedX = Math.fround(x),
      roundedY = Math.fround(y);
    const loss = Number(roundedX !== x) + Number(roundedY !== y);
    validateRequirement(
      id,
      precision === 'allow-loss' || loss === 0,
      `positions lose float64 precision at (${x}, ${y}); write precision: 'allow-loss' or supply local float32 coordinates`
    );
    roundedCoordinateCount += loss;
    const key = `${roundedX},${roundedY}`,
      original = `${x},${y}`;
    const originals = represented.get(key);
    if (!originals) represented.set(key, new Set([original]));
    else if (!originals.has(original)) {
      originals.add(original);
      collapsedPositionCount++;
    }
    positions.push(roundedX, roundedY);
  };
  const eachChild = (
    array: NativeArray,
    row: number,
    visit: (child: NativeArray, childRow: number) => void
  ) => {
    validateRequirement(
      id,
      array.kind === 'list' && row >= 0 && row < array.length && valid(array, row),
      'geometry nesting requires a valid list row'
    );
    if (array.kind !== 'list') return;
    const index = (array.offset ?? 0) + row,
      base = Number(array.offsetBase ?? 0);
    const first = Number(array.offsets[index]) - base,
      end = Number(array.offsets[index + 1]) - base;
    validateRequirement(
      id,
      Number.isSafeInteger(first) &&
        Number.isSafeInteger(end) &&
        first >= 0 &&
        end >= first &&
        end <= array.child.length,
      `list offsets must span valid child rows; received ${first}, ${end} for ${array.child.length}`
    );
    for (let childRow = first; childRow < end; childRow++) visit(array.child, childRow);
  };
  const appendLine = (array: NativeArray, row: number) => {
    eachChild(array, row, appendPosition);
    lineOffsets.push(positions.length / 2);
  };
  const appendRing = (array: NativeArray, row: number) => {
    eachChild(array, row, appendPosition);
    ringOffsets.push(positions.length / 2);
  };
  const appendPolygon = (array: NativeArray, row: number) => {
    eachChild(array, row, appendRing);
    polygonOffsets.push(ringOffsets.length - 1);
  };
  // The upstream adapter validates the Arrow schema, coordinate organization and list depth.
  for (const batch of iterateGeoArrowBatchesFromArrowVector(column, adapterOptions)) {
    const {encoding, dimension, edges} = batch.column;
    if (encoding !== 'geoarrow.point') singlePoints = false;
    validateRequirement(id, dimension === 'xy', `dimension must be xy; received ${dimension}`);
    validateRequirement(
      id,
      edges !== 'spherical',
      'spherical edges require projection before preparing planar geometry'
    );
    const nextKind =
      encoding === 'geoarrow.point' || encoding === 'geoarrow.multipoint'
        ? 'points'
        : encoding === 'geoarrow.linestring' || encoding === 'geoarrow.multilinestring'
          ? 'lines'
          : encoding === 'geoarrow.polygon' || encoding === 'geoarrow.multipolygon'
            ? 'polygons'
            : undefined;
    validateRequirement(
      id,
      Boolean(nextKind) && (!kind || kind === nextKind),
      `encoding must be native point, multipoint, linestring, multilinestring, polygon or multipolygon; received ${encoding}`
    );
    kind = nextKind;
    batchRowOffsets.push(batch.rowOffset);
    const array = batch.column.chunks[0];
    for (let row = 0; row < array.length; row++) {
      const sourceRow = batch.rowOffset + row,
        firstVertex = positions.length;
      if (valid(array, row)) {
        validity[sourceRow] = 1;
        switch (encoding) {
          case 'geoarrow.point':
            appendPosition(array, row, true);
            break;
          case 'geoarrow.multipoint':
            eachChild(array, row, (child, childRow) => appendPosition(child, childRow, true));
            break;
          case 'geoarrow.linestring':
            appendLine(array, row);
            break;
          case 'geoarrow.multilinestring':
            eachChild(array, row, appendLine);
            break;
          case 'geoarrow.polygon':
            appendPolygon(array, row);
            break;
          case 'geoarrow.multipolygon':
            eachChild(array, row, appendPolygon);
            break;
        }
        empty[sourceRow] = Number(positions.length === firstVertex);
      }
      featureOffsets.push(
        kind === 'points'
          ? positions.length / 2
          : kind === 'lines'
            ? lineOffsets.length - 1
            : polygonOffsets.length - 1
      );
      if (positions.length - firstVertex !== 2) singlePoints = false;
    }
  }
  batchRowOffsets.push(column.length);
  const sourceRows = Uint32Array.from({length: column.length}, (_, row) => rowOffset + row);
  const sourceIds = options.sourceIds
    ? new Uint32Array(options.sourceIds)
    : new Uint32Array(sourceRows);
  const geometry: CPUSpatialJoinGeometry = {
    kind: kind ?? 'points',
    positions: Float32Array.from(positions),
    ...(singlePoints ? {} : {featureOffsets: Uint32Array.from(featureOffsets)}),
    sourceIds
  };
  if (kind === 'lines') geometry.lineOffsets = Uint32Array.from(lineOffsets);
  if (kind === 'polygons')
    Object.assign(geometry, {
      polygonOffsets: Uint32Array.from(polygonOffsets),
      ringOffsets: Uint32Array.from(ringOffsets)
    });
  return {
    geometry,
    sourceRows,
    sourceIds,
    validity,
    empty,
    batchRowOffsets: Uint32Array.from(batchRowOffsets),
    frame: options.frame ? structuredClone(options.frame) : undefined,
    roundedCoordinateCount,
    collapsedPositionCount
  };
}
