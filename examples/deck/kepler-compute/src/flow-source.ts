// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type * as arrow from 'apache-arrow';
import type {FlowAggregateSource, FlowPairs} from './flow-gpu-aggregation';
import type {ClusterLevel, FlowAccessors, LocationAccessors} from '@flowmap.gl/data';

/** Native Flow layer LAT_LNG roles. Source coordinates are the published coordinate roles. */
export type FlowRoles = {
  targetLongitude: string;
  targetLatitude: string;
  count?: string;
  sourceName?: string;
  targetName?: string;
};
export type FlowLocation = {id: number; name: string; lon: number; lat: number};
export type FlowDatum = {index: number; sourceId: number; targetId: number; count: number};
export type FlowLayerData = {
  locations: FlowLocation[];
  flows: FlowDatum[];
  clusterLevels: ClusterLevel[];
};
export type PreparedFlowSource = {
  rowCount: number;
  locations: FlowLocation[];
  clusterLevels: ClusterLevel[];
  bounds: [number, number, number, number] | null;
  /** Retained CPU object storage estimate for memory admission (locations plus flow datums). */
  byteLength: number;
};
/** Per-row CPU source of the reference oracle `prepareFlowSource`. */
export type ReferenceFlowSource = PreparedFlowSource & {
  /** Location ID per source row, or 0 when the row has a non-finite endpoint coordinate. */
  sourceIds: Uint32Array;
  targetIds: Uint32Array;
  counts: Float64Array;
};

/** Pinned Kepler 3.3 flow-layer.js constants and accessors, reproduced for the same library. */
export const MAX_FLOW_CLUSTER_ZOOM_LEVEL = 20;
export const FLOW_LOCATION_ACCESSORS: LocationAccessors<FlowLocation> = {
  getLocationId: location => location.id,
  getLocationLon: location => location.lon,
  getLocationLat: location => location.lat,
  getLocationName: location => location.name
};
const FLOW_ROW_ACCESSORS = (
  sourceIds: Uint32Array,
  targetIds: Uint32Array,
  counts: Float64Array
): FlowAccessors<number> => ({
  getFlowOriginId: row => sourceIds[row],
  getFlowDestId: row => targetIds[row],
  getFlowMagnitude: row => counts[row]
});

/**
 * REFERENCE ORACLE (per-row CPU path, no longer used by the runtime): kept to test the
 * GPU-resident aggregation against native semantics.
 *
 * Mirrors native `updateLayerMeta`: locations are unique `lat:lon` strings in first-seen order
 * (source before target, ids from 1), names come from the first row that creates a location, and
 * clustering uses unfiltered row weights. Null or non-finite counts contribute zero; a missing
 * count column counts one. Rows with a non-finite endpoint keep their slot and never become flows.
 */
export async function prepareFlowSource(
  table: arrow.Table,
  coordinates: {longitude: string; latitude: string},
  roles: FlowRoles
): Promise<ReferenceFlowSource> {
  // The package uses extensionless ESM imports, so load it only where a bundler resolves them.
  const {clusterLocations, makeLocationWeightGetter} = await import('@flowmap.gl/data');
  const read = (name: string | undefined) => {
    const column = name ? table.getChild(name) : null;
    if (name && !column) throw new Error('Flow column is missing from the retained source');
    return column;
  };
  const sourceLongitude = read(coordinates.longitude)!;
  const sourceLatitude = read(coordinates.latitude)!;
  const targetLongitude = read(roles.targetLongitude)!;
  const targetLatitude = read(roles.targetLatitude)!;
  const countColumn = read(roles.count);
  const sourceNames = read(roles.sourceName);
  const targetNames = read(roles.targetName);
  const rowCount = table.numRows;
  const sourceIds = new Uint32Array(rowCount);
  const targetIds = new Uint32Array(rowCount);
  const counts = new Float64Array(rowCount);
  const locationsByKey = new Map<string, FlowLocation>();
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const addLocation = (lon: number, lat: number, name: unknown): number => {
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) return 0;
    const key = `${lat}:${lon}`;
    let location = locationsByKey.get(key);
    if (!location) {
      const id = locationsByKey.size + 1;
      location = {
        id,
        name: name === null || name === undefined ? `Location#${id}` : String(name),
        lat,
        lon
      };
      locationsByKey.set(key, location);
    }
    bounds[0] = Math.min(bounds[0], lon);
    bounds[1] = Math.min(bounds[1], lat);
    bounds[2] = Math.max(bounds[2], lon);
    bounds[3] = Math.max(bounds[3], lat);
    return location.id;
  };
  for (let row = 0; row < rowCount; row++) {
    const lon0: unknown = sourceLongitude.get(row);
    const lat0: unknown = sourceLatitude.get(row);
    const lon1: unknown = targetLongitude.get(row);
    const lat1: unknown = targetLatitude.get(row);
    // Native requires all four values finite before a row can be a flow.
    const finite = [lon0, lat0, lon1, lat1].every(
      value => typeof value === 'number' && Number.isFinite(value)
    );
    const sourceId = addLocation(lon0 as number, lat0 as number, sourceNames?.get(row));
    const targetId = addLocation(lon1 as number, lat1 as number, targetNames?.get(row));
    sourceIds[row] = finite ? sourceId : 0;
    targetIds[row] = finite ? targetId : 0;
    const count: unknown = countColumn ? countColumn.get(row) : 1;
    counts[row] = typeof count === 'number' && Number.isFinite(count) ? count : 0;
  }
  const locations = Array.from(locationsByKey.values());
  const getLocationWeight = makeLocationWeightGetter(
    Array.from({length: rowCount}, (_, row) => row),
    FLOW_ROW_ACCESSORS(sourceIds, targetIds, counts)
  );
  const clusterLevels = clusterLocations(locations, FLOW_LOCATION_ACCESSORS, getLocationWeight, {
    maxZoom: MAX_FLOW_CLUSTER_ZOOM_LEVEL
  });
  return {
    rowCount,
    locations,
    sourceIds,
    targetIds,
    counts,
    clusterLevels,
    bounds: bounds.every(Number.isFinite) ? bounds : null,
    byteLength:
      sourceIds.byteLength +
      targetIds.byteLength +
      counts.byteLength +
      locations.length * 128 +
      rowCount * 8
  };
}

/** REFERENCE ORACLE: native `calculateDataAttribute`; selected rows with both endpoints become flows. */
export function makeFlowData(
  source: ReferenceFlowSource,
  selected: Uint32Array | undefined
): FlowLayerData {
  const flows: FlowDatum[] = [];
  for (let row = 0; row < source.rowCount; row++) {
    if (selected && !selected[row]) continue;
    const sourceId = source.sourceIds[row];
    const targetId = source.targetIds[row];
    if (sourceId && targetId)
      flows.push({index: row, sourceId, targetId, count: source.counts[row]});
  }
  return {locations: source.locations, clusterLevels: source.clusterLevels, flows};
}

/** Reads one endpoint's exact coordinates and raw name from the retained Arrow table. */
export function makeFlowLocationReader(
  table: arrow.Table,
  coordinates: {longitude: string; latitude: string},
  roles: FlowRoles
): (endpoint: number) => {lon: number; lat: number; name: unknown} {
  const read = (name: string | undefined) => {
    const column = name ? table.getChild(name) : null;
    if (name && !column) throw new Error('Flow column is missing from the retained source');
    return column;
  };
  const sourceLongitude = read(coordinates.longitude)!;
  const sourceLatitude = read(coordinates.latitude)!;
  const targetLongitude = read(roles.targetLongitude)!;
  const targetLatitude = read(roles.targetLatitude)!;
  const sourceNames = read(roles.sourceName);
  const targetNames = read(roles.targetName);
  return endpoint => {
    const row = endpoint >> 1;
    const isTarget = (endpoint & 1) === 1;
    return {
      lon: (isTarget ? targetLongitude : sourceLongitude).get(row) as number,
      lat: (isTarget ? targetLatitude : sourceLatitude).get(row) as number,
      name: (isTarget ? targetNames : sourceNames)?.get(row)
    };
  };
}

/**
 * Builds locations in first-seen order from each location's first endpoint. Names come from the
 * row that created the location; null names read `Location#<id>`, like native.
 */
export function makeFlowLocations(
  firstEndpoints: Uint32Array,
  readEndpoint: ReturnType<typeof makeFlowLocationReader>
): FlowLocation[] {
  return Array.from(firstEndpoints, (endpoint, index) => {
    const {lon, lat, name} = readEndpoint(endpoint);
    const id = index + 1;
    return {
      id,
      name: name === null || name === undefined ? `Location#${id}` : String(name),
      lat,
      lon
    };
  });
}

/**
 * Prepares locations and clusters from GPU aggregates. Clustering stays with the pinned
 * `@flowmap.gl/data`, fed unfiltered aggregated origin-destination flows (their per-location
 * totals equal the native per-row totals).
 */
export async function prepareFlowSourceFromAggregates(
  table: arrow.Table,
  coordinates: {longitude: string; latitude: string},
  roles: FlowRoles,
  aggregate: Pick<FlowAggregateSource, 'rowCount' | 'firstEndpoints' | 'bounds'>,
  unfilteredPairs: FlowPairs
): Promise<PreparedFlowSource> {
  const {clusterLocations, makeLocationWeightGetter} = await import('@flowmap.gl/data');
  const locations = makeFlowLocations(
    aggregate.firstEndpoints,
    makeFlowLocationReader(table, coordinates, roles)
  );
  const getLocationWeight = makeLocationWeightGetter(
    Array.from(unfilteredPairs.sourceIds, (_, pair) => pair),
    {
      getFlowOriginId: pair => unfilteredPairs.sourceIds[pair],
      getFlowDestId: pair => unfilteredPairs.targetIds[pair],
      getFlowMagnitude: pair => unfilteredPairs.counts[pair]
    }
  );
  const clusterLevels = clusterLocations(locations, FLOW_LOCATION_ACCESSORS, getLocationWeight, {
    maxZoom: MAX_FLOW_CLUSTER_ZOOM_LEVEL
  });
  return {
    rowCount: aggregate.rowCount,
    locations,
    clusterLevels,
    bounds: aggregate.bounds,
    byteLength: locations.length * 256 + unfilteredPairs.sourceIds.length * 64
  };
}

/** One datum per aggregated origin-destination pair (representative row as `index`). */
export function makeFlowDataFromPairs(source: PreparedFlowSource, pairs: FlowPairs): FlowLayerData {
  const flows: FlowDatum[] = new Array(pairs.sourceIds.length);
  for (let pair = 0; pair < flows.length; pair++)
    flows[pair] = {
      index: pairs.firstRows[pair],
      sourceId: pairs.sourceIds[pair],
      targetId: pairs.targetIds[pair],
      count: pairs.counts[pair]
    };
  return {locations: source.locations, clusterLevels: source.clusterLevels, flows};
}
