// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/graph/arrow';
import {describeWideNumericNarrowing} from './analysis2-narrowing-summary';
import {previewArrowSource} from './source-preview';
import {readFieldRoute, routeArrowTable} from './source-type-routes';
import {preparePublishedGeometry, preparePublishedH3} from './source-publication';
import {appendDGGSFootprintColumn, DGGS_FOOTPRINT_COLUMN, type DGGSFamily} from './dggs-source';
import {prepareSandboxTrips, type TripPreparationOptions} from './trip-source-options';
import type {CoordinateRoles} from './point-position-source';
import type {NativeCellIndex} from './cell-layer-types';

/**
 * Default Float64 policy for automatic preparation: round to Float32 and measure the error. The
 * relative budget is the Float32 unit roundoff (2^-24) doubled, and the tiny absolute budget only
 * covers values below the smallest normal Float32. Values that overflow Float32 still fail and
 * their column is excluded, never silently clamped.
 */
export const AUTO_PREPARE_FLOAT64_POLICY = Object.freeze({
  mode: 'round' as const,
  maximumAbsoluteError: 1e-37,
  maximumRelativeError: 2 ** -23
});

/** Trip time budgets tried in order; the first that admits every timed vertex is recorded. */
export const AUTO_PREPARE_TRIP_ERROR_BUDGETS_MILLISECONDS = [0, 1, 1000] as const;

/** One layer's column roles, independent of Kepler layer classes so the plan is testable in node. */
export type LayerRoleInput = {
  id: string;
  type: string;
  isVisible?: boolean;
  /** Roles pinned by a share link: win over Kepler's guessed layers regardless of type priority. */
  preferred?: boolean;
  columnMode?: string;
  /** Column role name to field name (Kepler `config.columns[role].value`). */
  columns: Readonly<Record<string, string | null | undefined>>;
};

export type FieldRoleInput = {name: string; type: string};

export type AutoPreparePlan = {
  /** Source table, with a derived footprint column appended for S2/geohash/A5 layers. */
  table: arrow.Table;
  coordinates?: CoordinateRoles;
  geometryColumn?: string;
  h3Column?: string;
  cellIndex?: NativeCellIndex;
  tripOptions?: TripPreparationOptions;
  normalization: PrepareArrowTableForGPUAnalyticsOptions;
  /** Columns handled outside scalar normalization (geometry, H3, cell IDs). */
  sourceOnlyColumns: readonly string[];
  /** Human-readable role decisions, e.g. "Longitude/latitude: lng, lat (from layer Points)". */
  roleSummary: readonly string[];
  /** Human-readable precision policy decisions, always populated. */
  policySummary: readonly string[];
  excluded: readonly {name: string; reason: string}[];
  /** Non-fatal problems such as a rejected role or a layer needing unprepared columns. */
  warnings: readonly string[];
  /**
   * Layers on a second timed trip column of the same dataset. A publication carries one trip role
   * set, so these layers are kept but hidden. Geometry, H3 and cell layers on other columns render
   * through on-demand geometry column variants instead.
   */
  unpreparedLayerIds: readonly string[];
};

const POINT_TYPES = new Set(['point', 'icon', '3D', 'heatmap', 'hexagon', 'grid', 'cluster']);
const CONNECTION_TYPES = new Set(['arc', 'line']);
const CELL_FAMILIES: Readonly<Record<string, DGGSFamily>> = {
  s2: 's2',
  geohash: 'geohash',
  a5: 'a5'
};
/** Highest priority first: a publication carries exactly one geometry role set. */
const ROLE_PRIORITY = [
  'trip',
  'geojson',
  'hexagonId',
  's2',
  'a5',
  'geohash',
  'flow',
  'point',
  'icon',
  '3D',
  'heatmap',
  'hexagon',
  'grid',
  'cluster',
  'arc',
  'line'
];

type RoleChoice = {
  layer: LayerRoleInput;
  family: 'trip' | 'geojson' | 'h3' | 'cell' | 'point' | 'flow' | 'connection';
};

function readColumn(layer: LayerRoleInput, role: string, table: arrow.Table): string | undefined {
  const name = layer.columns[role];
  return name && table.schema.fields.some(field => field.name === name) ? name : undefined;
}

function classifyLayer(layer: LayerRoleInput, table: arrow.Table): RoleChoice | undefined {
  if (layer.type === 'trip') {
    if (layer.columnMode === 'table') {
      return ['id', 'lat', 'lng', 'timestamp'].every(role => readColumn(layer, role, table))
        ? {layer, family: 'trip'}
        : undefined;
    }
    return readColumn(layer, 'geojson', table) ? {layer, family: 'trip'} : undefined;
  }
  if (layer.type === 'geojson')
    return readColumn(layer, 'geojson', table) ? {layer, family: 'geojson'} : undefined;
  if (layer.type === 'hexagonId')
    return readColumn(layer, 'hex_id', table) ? {layer, family: 'h3'} : undefined;
  if (layer.type in CELL_FAMILIES)
    return readColumn(layer, 'token', table) ? {layer, family: 'cell'} : undefined;
  if (layer.type === 'flow')
    return layer.columnMode !== 'H3' &&
      ['lat0', 'lng0', 'lat1', 'lng1'].every(role => readColumn(layer, role, table))
      ? {layer, family: 'flow'}
      : undefined;
  if (POINT_TYPES.has(layer.type))
    return readColumn(layer, 'lat', table) && readColumn(layer, 'lng', table)
      ? {layer, family: 'point'}
      : undefined;
  if (CONNECTION_TYPES.has(layer.type))
    return ['lat0', 'lng0', 'lat1', 'lng1'].every(role => readColumn(layer, role, table))
      ? {layer, family: 'connection'}
      : undefined;
  return undefined;
}

/** Fields the Kepler analyzer typed as geometry, H3 or point pairs, used when no layer exists. */
export function inferLayerRolesFromFields(
  fields: readonly FieldRoleInput[],
  findPointFieldPairs: (fields: readonly FieldRoleInput[]) => readonly {
    pair: {lat: {value: string}; lng: {value: string}; altitude?: {value: string}};
  }[]
): LayerRoleInput[] {
  const layers: LayerRoleInput[] = [];
  for (const field of fields)
    if (field.type === 'geojson' || field.type === 'geoarrow')
      layers.push({
        id: `inferred-geojson-${field.name}`,
        type: 'geojson',
        columns: {geojson: field.name}
      });
  for (const field of fields)
    if (field.type === 'h3')
      layers.push({
        id: `inferred-h3-${field.name}`,
        type: 'hexagonId',
        columns: {hex_id: field.name}
      });
  for (const {pair} of findPointFieldPairs(fields))
    layers.push({
      id: `inferred-point-${pair.lat.value}`,
      type: 'point',
      columns: {
        lat: pair.lat.value,
        lng: pair.lng.value,
        ...(pair.altitude ? {altitude: pair.altitude.value} : {})
      }
    });
  return layers;
}

/** Epoch seconds, epoch milliseconds or ISO text, sampled from the first timed vertices. */
export function inferTripTimestampEncoding(
  table: arrow.Table,
  column: string
): 'unix-second' | 'unix-millisecond' | 'iso8601' | undefined {
  const vector = table.getChild(column);
  if (!vector) return undefined;
  const sampleLimit = Math.min(table.numRows, 64);
  for (let row = 0; row < sampleLimit; row++) {
    const value: unknown = vector.get(row);
    if (typeof value !== 'string') continue;
    let feature: unknown;
    try {
      feature = JSON.parse(value);
    } catch {
      continue;
    }
    const geometry =
      feature && typeof feature === 'object' && 'geometry' in feature
        ? (feature as {geometry: unknown}).geometry
        : feature;
    const coordinates =
      geometry && typeof geometry === 'object' && 'coordinates' in geometry
        ? (geometry as {coordinates: unknown}).coordinates
        : undefined;
    const vertex = Array.isArray(coordinates) ? coordinates[0] : undefined;
    const time: unknown = Array.isArray(vertex) ? vertex[3] : undefined;
    if (typeof time === 'string') return 'iso8601';
    if (typeof time === 'number' && Number.isFinite(time))
      return Math.abs(time) < 1e11 ? 'unix-second' : 'unix-millisecond';
  }
  return undefined;
}

function describeConversion(table: arrow.Table, name: string): string {
  const field = table.schema.fields.find(candidate => candidate.name === name);
  return field ? String(field.type) : 'unknown';
}

/**
 * Chooses roles from the native layers (or inferred layers), validates them on the CPU, and
 * selects every supported field with the default precision policy. Never touches the GPU.
 * Roles that fail CPU validation degrade to a table with a warning instead of failing the load.
 */
export function makeAutoPreparePlan(
  inputTable: arrow.Table,
  layers: readonly LayerRoleInput[]
): AutoPreparePlan {
  // Type routing keeps every original column and converts or excludes unsupported Arrow types.
  const sourceTable = routeArrowTable(inputTable).table;
  const warnings: string[] = [];
  const unpreparedLayerIds: string[] = [];
  const roleSummary: string[] = [];
  const policySummary: string[] = [];
  let table = sourceTable;
  let coordinates: CoordinateRoles | undefined;
  let geometryColumn: string | undefined;
  let h3Column: string | undefined;
  let cellIndex: NativeCellIndex | undefined;
  let tripOptions: TripPreparationOptions | undefined;

  const candidates = layers
    .map(layer => classifyLayer(layer, sourceTable))
    .filter((choice): choice is RoleChoice => Boolean(choice))
    .sort((left, right) => {
      const visible =
        Number(right.layer.isVisible !== false) - Number(left.layer.isVisible !== false);
      return (
        Number(Boolean(right.layer.preferred)) - Number(Boolean(left.layer.preferred)) ||
        visible ||
        ROLE_PRIORITY.indexOf(left.layer.type) - ROLE_PRIORITY.indexOf(right.layer.type)
      );
    });
  const choice = candidates[0];
  if (choice) {
    const {layer} = choice;
    const columns = layer.columns as Record<string, string | undefined>;
    try {
      switch (choice.family) {
        case 'trip': {
          if (layer.columnMode === 'table') {
            geometryColumn = columns['id'];
            // Grouped tabular trips: the ID column is the geometry-role column.
            tripOptions = {
              timestampEncoding: 'unix-millisecond',
              maximumTimeErrorMilliseconds: 0,
              grouping: {
                timestampColumn: columns['timestamp']!,
                longitudeColumn: columns['lng']!,
                latitudeColumn: columns['lat']!,
                ...(columns['altitude'] ? {altitudeColumn: columns['altitude']} : {})
              }
            } as TripPreparationOptions;
          } else {
            geometryColumn = columns['geojson'];
            const encoding = inferTripTimestampEncoding(sourceTable, geometryColumn!);
            if (!encoding) throw new Error('No timestamps found in the first path vertices');
            tripOptions = {
              timestampEncoding: encoding,
              maximumTimeErrorMilliseconds: 0
            } as TripPreparationOptions;
          }
          let accepted = false;
          let lastFailure = '';
          for (const budget of AUTO_PREPARE_TRIP_ERROR_BUDGETS_MILLISECONDS) {
            tripOptions = {
              ...tripOptions,
              maximumTimeErrorMilliseconds: budget
            } as TripPreparationOptions;
            try {
              prepareSandboxTrips(sourceTable, geometryColumn!, tripOptions);
              accepted = true;
              break;
            } catch (failure) {
              lastFailure = failure instanceof Error ? failure.message : String(failure);
            }
          }
          if (!accepted) throw new Error(lastFailure);
          roleSummary.push(
            `Trips: ${geometryColumn} from layer ${layer.id} (${tripOptions.timestampEncoding}, time budget ${tripOptions.maximumTimeErrorMilliseconds} ms)`
          );
          policySummary.push(
            tripOptions.maximumTimeErrorMilliseconds === 0
              ? 'Trip times: exact relative Float32 milliseconds'
              : `Trip times: relative Float32 milliseconds with an explicit ${tripOptions.maximumTimeErrorMilliseconds} ms budget`
          );
          break;
        }
        case 'geojson':
          geometryColumn = columns['geojson'];
          preparePublishedGeometry(sourceTable, geometryColumn!);
          roleSummary.push(`Geometry: ${geometryColumn} from layer ${layer.id}`);
          policySummary.push('Geometry: Float32 planar positions; original geometry retained');
          break;
        case 'h3':
          h3Column = columns['hex_id'];
          preparePublishedH3(sourceTable, h3Column!);
          roleSummary.push(`H3 cells: ${h3Column} from layer ${layer.id}`);
          policySummary.push('H3: exact cell IDs; Float32 boundary and center coordinates');
          break;
        case 'cell': {
          const family = CELL_FAMILIES[layer.type];
          const cellColumn = columns['token']!;
          const appended = appendDGGSFootprintColumn(sourceTable, cellColumn, family);
          table = appended.table;
          geometryColumn = DGGS_FOOTPRINT_COLUMN;
          cellIndex = {family, column: cellColumn} as NativeCellIndex;
          roleSummary.push(
            `${family.toUpperCase()} cells: ${cellColumn} from layer ${layer.id} (${appended.summary.validRowCount} valid)`
          );
          policySummary.push(
            `${family.toUpperCase()}: exact IDs; derived Float32 footprints retain source rows`
          );
          break;
        }
        case 'flow':
          coordinates = {
            longitude: columns['lng0']!,
            latitude: columns['lat0']!,
            flow: {
              targetLongitude: columns['lng1']!,
              targetLatitude: columns['lat1']!,
              ...(columns['count'] ? {count: columns['count']} : {}),
              ...(columns['sourceName'] ? {sourceName: columns['sourceName']} : {}),
              ...(columns['targetName'] ? {targetName: columns['targetName']} : {})
            }
          };
          roleSummary.push(
            `Flow: ${columns['lng0']}/${columns['lat0']} to ${columns['lng1']}/${columns['lat1']} from layer ${layer.id}`
          );
          policySummary.push('Coordinates: Float32 longitude/latitude degrees');
          break;
        case 'connection':
          coordinates = {
            longitude: columns['lng0']!,
            latitude: columns['lat0']!,
            // Native Line reads optional alt0/alt1; Arc has no altitude columns.
            ...(layer.type === 'line' && columns['alt0'] && table.getChild(columns['alt0'])
              ? {altitude: columns['alt0']}
              : {}),
            ...(columns['lng1'] && columns['lat1']
              ? {
                  target: {
                    longitude: columns['lng1'],
                    latitude: columns['lat1'],
                    ...(layer.type === 'line' && columns['alt1'] && table.getChild(columns['alt1'])
                      ? {altitude: columns['alt1']}
                      : {})
                  }
                }
              : {})
          };
          roleSummary.push(
            `Connection origins: ${coordinates.longitude}/${coordinates.latitude} from layer ${layer.id}`
          );
          policySummary.push('Coordinates: Float32 longitude/latitude degrees');
          break;
        case 'point':
          coordinates = {
            longitude: columns['lng']!,
            latitude: columns['lat']!,
            ...(columns['altitude'] && table.getChild(columns['altitude'])
              ? {altitude: columns['altitude']}
              : {})
          };
          roleSummary.push(
            `Longitude/latitude: ${coordinates.longitude}/${coordinates.latitude}${coordinates.altitude ? ` (altitude ${coordinates.altitude})` : ''} from layer ${layer.id}`
          );
          policySummary.push('Coordinates: Float32 longitude/latitude degrees');
          break;
      }
    } catch (failure) {
      coordinates = undefined;
      geometryColumn = undefined;
      h3Column = undefined;
      cellIndex = undefined;
      tripOptions = undefined;
      table = sourceTable;
      roleSummary.length = 0;
      policySummary.length = 0;
      warnings.push(
        `Layer ${layer.id} (${layer.type}) was prepared as a table because its columns were rejected: ${failure instanceof Error ? failure.message : String(failure)}`
      );
    }
    for (const other of candidates.slice(1)) {
      // Geometry/H3/cell layers on another column or family of the same dataset are prepared on
      // demand as geometry column variants (geometry-column-variants.ts), so they all render.
      if (other.family === 'geojson' || other.family === 'h3' || other.family === 'cell') continue;
      if (
        other.family === choice.family &&
        other.family === 'trip' &&
        JSON.stringify(other.layer.columns) !== JSON.stringify(choice.layer.columns)
      ) {
        unpreparedLayerIds.push(other.layer.id);
        warnings.push(
          `Layer ${other.layer.id} (${other.layer.type}) uses a different trip column than the prepared one; it is hidden because a dataset publishes one timed trip column.`
        );
        continue;
      }
      if (other.family === choice.family && other.layer.type === choice.layer.type) continue;
      warnings.push(
        `Layer ${other.layer.id} (${other.layer.type}) uses columns that are not the prepared position role.`
      );
    }
  } else if (layers.length) {
    warnings.push('No layer has a position, geometry, H3, cell or trip role; prepared as a table.');
  } else {
    roleSummary.push('No geometry or coordinate fields found; prepared as a table.');
  }

  if (!geometryColumn && !coordinates && !h3Column && !cellIndex) {
    // A WKT column routed to derived GeoJSON becomes the geometry role when no layer claims one.
    const derived = table.schema.fields.find(
      field => readFieldRoute(field)?.kind === 'derived-geometry' && field.name.endsWith('_geojson')
    );
    if (derived)
      try {
        preparePublishedGeometry(table, derived.name);
        geometryColumn = derived.name;
        roleSummary.push(`Geometry: ${derived.name} (derived from WKT text)`);
        policySummary.push('Geometry: Float32 planar positions; original WKT retained');
      } catch (failure) {
        warnings.push(
          `Derived WKT geometry ${derived.name} was rejected: ${failure instanceof Error ? failure.message : String(failure)}`
        );
      }
  }
  const sourceOnlyColumns = [geometryColumn, h3Column, cellIndex?.column].filter(
    (name): name is string => Boolean(name)
  );
  const normalization: PrepareArrowTableForGPUAnalyticsOptions = {
    float64: AUTO_PREPARE_FLOAT64_POLICY
  };
  policySummary.unshift(
    `Float64 fields: round to Float32, relative error up to ${AUTO_PREPARE_FLOAT64_POLICY.maximumRelativeError.toExponential(2)} (measured per field in the schema preview)`
  );
  policySummary.push(...describeWideNumericNarrowing(table));
  const roleColumns = new Set(
    [
      coordinates?.longitude,
      coordinates?.latitude,
      coordinates?.altitude,
      coordinates?.flow?.targetLongitude,
      coordinates?.flow?.targetLatitude,
      coordinates?.target?.longitude,
      coordinates?.target?.latitude,
      coordinates?.target?.altitude
    ].filter(Boolean)
  );
  const preview = previewArrowSource(table, normalization, sourceOnlyColumns);
  const excluded: {name: string; reason: string}[] = [];
  const columns: string[] = [];
  for (const field of table.schema.fields) {
    const route = readFieldRoute(field);
    const isCommon =
      !route ||
      route.kind === 'gpu-exact' ||
      route.kind === 'gpu-dictionary' ||
      route.sourceType.startsWith('Float64') ||
      route.sourceType.startsWith('Timestamp');
    if (route && !isCommon)
      policySummary.push(`Type route ${field.name} (${route.sourceType}): ${route.route}`);
  }
  for (const field of preview.fields) {
    if (field.kind === 'unsupported') {
      const reason = field.reason ?? `Unsupported type ${describeConversion(table, field.name)}`;
      excluded.push({name: field.name, reason});
      if (roleColumns.has(field.name)) {
        // A position role without its GPU column cannot render; degrade to a table visibly.
        warnings.push(
          `Coordinate field ${field.name} cannot be prepared (${reason}); prepared as a table.`
        );
        coordinates = undefined;
      }
    } else columns.push(field.name);
  }
  if (excluded.length)
    policySummary.push(
      `Excluded from GPU upload (original values stay in the retained source): ${excluded.map(entry => entry.name).join(', ')}`
    );
  return {
    table,
    ...(coordinates ? {coordinates} : {}),
    ...(geometryColumn ? {geometryColumn} : {}),
    ...(h3Column ? {h3Column} : {}),
    ...(cellIndex ? {cellIndex} : {}),
    ...(tripOptions ? {tripOptions} : {}),
    normalization: {...normalization, columns},
    sourceOnlyColumns,
    roleSummary,
    policySummary,
    excluded,
    warnings,
    unpreparedLayerIds
  };
}
