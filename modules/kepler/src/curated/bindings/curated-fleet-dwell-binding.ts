// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {
  addFleetDwellRecipe,
  getGPUTrajectoryMetricsParameterValues,
  type GPUFleetDwellRecipeResult
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {
  appendCatalogRow,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogFeatureCollection
} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {readRecipePolygons} from '../../analysis/catalog-platform5-recipe-common';
import {
  MAXIMUM_TRAJECTORY_FIXES,
  MAXIMUM_TRAJECTORY_TRACKS,
  TRAJECTORY_PARAMETERS,
  numberParameter,
  prepareCatalogTrajectory,
  makeTrajectoryOutput
} from '../../analysis/catalog-trajectory-common';

const BINDING_ID = 'curated-fleet-dwell';
const NO_ZONE = 0xffffffff;
export const FLEET_DWELL_LIMITS = {
  fixes: MAXIMUM_TRAJECTORY_FIXES,
  tracks: MAXIMUM_TRAJECTORY_TRACKS,
  zones: 128,
  zoneVertices: 16384,
  stops: MAXIMUM_TRAJECTORY_FIXES,
  candidates: MAXIMUM_TRAJECTORY_FIXES * 128
} as const;

const PARAMETERS = [
  ...TRAJECTORY_PARAMETERS,
  numberParameter('stopSpeedThreshold', 'Stop speed threshold', 1, 0, 1e6, 'crs/s'),
  numberParameter('stopMinimumDuration', 'Minimum stop duration', 20, 0, 1e9, 'seconds'),
  {
    name: 'stopCapacity',
    label: 'Stop capacity',
    kind: 'integer' as const,
    minimum: 1,
    maximum: FLEET_DWELL_LIMITS.stops,
    defaultValue: FLEET_DWELL_LIMITS.stops
  },
  {
    name: 'candidateCapacity',
    label: 'Stop / zone candidate capacity',
    kind: 'integer' as const,
    minimum: 1,
    maximum: FLEET_DWELL_LIMITS.candidates,
    defaultValue: FLEET_DWELL_LIMITS.candidates
  }
];

/** One preparation, shared with the existing trajectory bindings; times remain signed Int64 ms. */
export function prepareFleetDwell(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  overrides: Readonly<Record<string, CatalogParameterValue>>
) {
  const values: Record<string, CatalogParameterValue> = Object.fromEntries(
    PARAMETERS.map(parameter => [parameter.name, parameter.defaultValue!])
  );
  Object.assign(values, overrides);
  for (const parameter of PARAMETERS) {
    const value = values[parameter.name];
    if (parameter.kind === 'number' || parameter.kind === 'integer') {
      validateFiniteNumber(
        BINDING_ID,
        parameter.name,
        value as number,
        parameter.minimum,
        parameter.maximum
      );
      if (parameter.kind === 'integer')
        validateRequirement(
          BINDING_ID,
          Number.isInteger(value),
          `${parameter.name} must be an integer`
        );
    } else if (parameter.options) {
      validateRequirement(
        BINDING_ID,
        parameter.options.includes(String(value)),
        `${parameter.name} must be one of ${parameter.options.join(', ')}`
      );
    } else {
      validateRequirement(
        BINDING_ID,
        typeof value === 'string' && value.length <= 128,
        `${parameter.name} must be a string of at most 128 characters`
      );
    }
  }
  validateRequirement(
    BINDING_ID,
    Boolean(inputs['tracks'] && inputs['zones']),
    'tracks and zones inputs are required'
  );
  const trajectory = prepareCatalogTrajectory(inputs['tracks'], values);
  const collection = inputs['zones'];
  validateCapacity(
    BINDING_ID,
    'zones.length',
    collection.features.length,
    1,
    FLEET_DWELL_LIMITS.zones
  );
  // Reject malformed coordinates before the common packer can discard non-position entries.
  const validZones = collection.features.every(({geometry}) => {
    if (!geometry || !['Polygon', 'MultiPolygon'].includes(geometry.type)) return false;
    const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
    return (
      Array.isArray(polygons) &&
      polygons.length > 0 &&
      polygons.every(
        polygon =>
          Array.isArray(polygon) &&
          polygon.length > 0 &&
          polygon.every(
            ring =>
              Array.isArray(ring) &&
              ring.length >= 3 &&
              ring.every(
                position =>
                  Array.isArray(position) &&
                  position.length >= 2 &&
                  typeof position[0] === 'number' &&
                  Number.isFinite(position[0]) &&
                  typeof position[1] === 'number' &&
                  Number.isFinite(position[1]) &&
                  (values['coordinateSystem'] !== 'EPSG:4326' ||
                    (Math.abs(position[0]) <= 180 && Math.abs(position[1]) <= 90))
              )
          )
      )
    );
  });
  validateRequirement(
    BINDING_ID,
    validZones,
    'zones must contain finite Polygon or MultiPolygon rings with at least three positions in the declared coordinate system'
  );
  const parts = collection.features.map(feature => getPolygonParts(feature.geometry));
  const positions = parts.flat(3);
  validateCapacity(
    BINDING_ID,
    'zones.vertexCount',
    positions.length,
    3,
    FLEET_DWELL_LIMITS.zoneVertices
  );
  const zones = readRecipePolygons(collection, trajectory.frame);
  validateRequirement(
    BINDING_ID,
    zones.positions.every(Number.isFinite),
    'zones must project to finite Float32 positions'
  );
  return {
    values,
    trajectory,
    zones,
    collection,
    stopCapacity: Number(values['stopCapacity']),
    candidateCapacity: Number(values['candidateCapacity']),
    metricParameters: getGPUTrajectoryMetricsParameterValues({
      stopSpeedThreshold: Number(values['stopSpeedThreshold']) / 1000,
      stopMinimumDuration: Number(values['stopMinimumDuration']) * 1000
    })
  };
}

export type FleetDwellReadback = {
  count: number;
  requiredCount: number;
  stopOverflow: number;
  joinOverflow: number;
  joinUncertainCount: number;
  tableOverflow: number;
  tableCount: number;
  tracks: Uint32Array;
  startRows: Uint32Array;
  endRows: Uint32Array;
  centroids: Float32Array;
  durations: Float32Array;
  stopZones: Uint32Array;
  counts: Uint32Array;
  totals: Float32Array;
  means: Float32Array;
  maximums: Float32Array;
};

/** Recipe glue has no contributor ID; report only the named analysis stages. */
export function getFleetDwellContributorNames(
  contributors: GPUFleetDwellRecipeResult['contributors']
): string[] {
  return contributors
    .filter(contributor => 'id' in contributor)
    .map(contributor => contributor.constructor.name);
}

/** Refuse truncated or uncertain aggregates before building any publishable rows. */
export function makeFleetDwellOutput(
  prepared: ReturnType<typeof prepareFleetDwell>,
  result: FleetDwellReadback
): CatalogRunOutput {
  const {trajectory, zones, collection} = prepared;
  if (
    result.stopOverflow ||
    result.joinOverflow ||
    result.tableOverflow ||
    result.requiredCount > prepared.stopCapacity
  )
    throw new Error(
      `${BINDING_ID}: capacity overflow (stops ${result.requiredCount}/${prepared.stopCapacity}, stopOverflow=${result.stopOverflow}, joinOverflow=${result.joinOverflow}, tableOverflow=${result.tableOverflow}); increase stopCapacity/candidateCapacity or filter the input; no partial result published.`
    );
  // The recipe joins every allocated stop slot, including padding. This is a candidate count,
  // not an exact number of real uncertain stops. Conservative refusal avoids false precision.
  if (result.joinUncertainCount)
    throw new Error(
      `${BINDING_ID}: joinUncertainCount=${result.joinUncertainCount} uncertain stop-slot / zone candidates (including padding); simplify or rescale zones before retrying; no definitive dwell totals published.`
    );
  validateCapacity(BINDING_ID, 'stopCount', result.count, 0, prepared.stopCapacity);
  validateRequirement(
    BINDING_ID,
    result.count === result.requiredCount && result.tableCount === zones.featureCount,
    'GPU stop and zone counts must match complete output capacities'
  );
  const stops = makeCatalogTable(
    'fleet-stops',
    'Vehicle stops',
    {
      stop_index: 'int32',
      track_id: 'string',
      source_index: 'int32',
      vertex_index: 'int32',
      end_source_index: 'int32',
      end_vertex_index: 'int32',
      start_time: 'float64',
      end_time: 'float64',
      duration_seconds: 'float64',
      zone_index: 'int32',
      zone_id: 'string',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  let unassignedStopCount = 0;
  for (let stop = 0; stop < result.count; stop++) {
    const first = trajectory.fixes[result.startRows[stop]];
    const last = trajectory.fixes[result.endRows[stop]];
    const zone = result.stopZones[stop];
    if (zone === NO_ZONE) unassignedStopCount++;
    appendCatalogRow(stops, {
      stop_index: stop,
      track_id: trajectory.trackIds[result.tracks[stop]],
      source_index: first.sourceIndex,
      vertex_index: first.vertexIndex,
      end_source_index: last.sourceIndex,
      end_vertex_index: last.vertexIndex,
      start_time: first.timestamp,
      end_time: last.timestamp,
      duration_seconds: result.durations[stop] / 1000,
      zone_index: zone === NO_ZONE ? null : zones.sourceIndices[zone],
      zone_id:
        zone === NO_ZONE
          ? null
          : String(collection.features[zones.sourceIndices[zone]].id ?? zones.sourceIndices[zone]),
      status: zone === NO_ZONE ? 'outside-zones' : 'assigned',
      geometry: makeCatalogGeometryText(
        'Point',
        trajectory.frame.inverse(result.centroids[stop * 2], result.centroids[stop * 2 + 1])
      )
    });
  }
  const dwell = makeCatalogTable(
    'fleet-zone-dwell',
    'Dwell by zone',
    {
      zone_index: 'int32',
      zone_id: 'string',
      stop_count: 'int32',
      total_dwell_seconds: 'float64',
      mean_dwell_seconds: 'float64',
      maximum_dwell_seconds: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (let zone = 0; zone < zones.featureCount; zone++) {
    const sourceIndex = zones.sourceIndices[zone];
    const feature = collection.features[sourceIndex];
    appendCatalogRow(dwell, {
      zone_index: sourceIndex,
      zone_id: String(feature.id ?? sourceIndex),
      stop_count: result.counts[zone],
      total_dwell_seconds: result.totals[zone] / 1000,
      mean_dwell_seconds: result.counts[zone] ? result.means[zone] / 1000 : null,
      maximum_dwell_seconds: result.counts[zone] ? result.maximums[zone] / 1000 : null,
      geometry: JSON.stringify(feature.geometry)
    });
  }
  const output = makeTrajectoryOutput(trajectory, [stops, dwell], {
    stopCount: result.count,
    unassignedStopCount,
    zoneCount: zones.featureCount,
    stopOverflow: result.stopOverflow,
    joinOverflow: result.joinOverflow,
    joinUncertainCount: result.joinUncertainCount,
    tableOverflow: result.tableOverflow,
    requiredStopCount: result.requiredCount,
    totalAssignedDwellSeconds: result.totals.reduce((total, value) => total + value, 0) / 1000
  });
  output.diagnostics.push({
    code: 'fleet-dwell-sampling',
    severity: 'warning',
    count: result.count,
    message:
      'Stops use sampled slow-step runs and arithmetic centroids. Observation endpoints censor dwell; gaps are not interpolated or capped, and purpose is unknown.'
  });
  if (unassignedStopCount)
    output.diagnostics.push({
      code: 'fleet-dwell-unassigned',
      severity: 'warning',
      count: unassignedStopCount,
      message:
        'Stops outside the half-open zones remain in the stop table and are excluded from zone totals.'
    });
  return output;
}

export const CURATED_FLEET_DWELL_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Fleet dwell',
  domain: 'analysis',
  summary: 'Find slow-step stops, join their centroids to zones and summarize stop dwell per zone.',
  capabilityIds: ['movement-analysis'],
  contributorNames: ['GPUTrajectoryMetrics', 'GPUPointInPolygonJoin', 'GPUGroupStatistics'],
  gpuContributorNames: ['GPUTrajectoryMetrics', 'GPUPointInPolygonJoin', 'GPUGroupStatistics'],
  engine: 'gpu',
  inputs: [
    {
      name: 'tracks',
      label: 'Point fixes or timed tracks',
      accepts: ['point', 'line'],
      maximumFeatures: FLEET_DWELL_LIMITS.fixes,
      maximumVertices: FLEET_DWELL_LIMITS.fixes
    },
    {
      name: 'zones',
      label: 'Dwell zones',
      accepts: ['polygon'],
      maximumFeatures: FLEET_DWELL_LIMITS.zones,
      maximumVertices: FLEET_DWELL_LIMITS.zoneVertices
    }
  ],
  parameters: PARAMETERS,
  outputs: [
    {
      tableId: 'fleet-stops',
      label: 'Vehicle stops',
      geometry: 'point',
      identityColumns: ['stop_index', 'track_id', 'source_index', 'vertex_index'],
      maximumRows: FLEET_DWELL_LIMITS.stops
    },
    {
      tableId: 'fleet-zone-dwell',
      label: 'Dwell by zone',
      geometry: 'polygon',
      identityColumns: ['zone_index', 'zone_id'],
      maximumRows: FLEET_DWELL_LIMITS.zones
    }
  ],
  cleanup:
    'The catalog GPU session destroys its compiled graph and buffers in finally; plain output tables publish through native Sources.',
  declaredBehavior: [
    'Reuse trajectory preparation: stable track/time sorting, signed Int64 epoch milliseconds, original source and vertex identities. Requires distinct increasing times per track; null geometries are skipped with a diagnostic.',
    'Speed threshold is length/second (metres for EPSG:4326, declared planar units otherwise); minimum duration and all reported dwell are seconds. The GPU receives length/millisecond and milliseconds.',
    'Geographic tracks and zones share the trajectory local azimuthal equidistant frame. Planar coordinates share the shifted source frame. Holes and multipart zones are preserved; smallest containing feature row wins overlaps under half-open boundaries.',
    'A stop is a maximal slow-step run of sufficient duration, assigned by its arithmetic centroid. Stops outside zones remain visible. Every zone is retained, including empty zones with zero total and null mean/maximum.',
    'Stop, join or aggregate overflow rejects publication. Any uncertain join candidate rejects definitive totals; the recipe counts allocated padding slots too. No partial output is returned.'
  ],
  limitation:
    'GPU only. Two-dimensional sampled stop dwell, not all time inside zones. Long gaps count toward a slow run; split or clean tracks first. Observation endpoints censor dwell. Local projection and Float32 positions/aggregates are approximate; no stop-purpose inference.',
  async runGpu(context, inputs, values) {
    const prepared = prepareFleetDwell(inputs, values);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const capacity = prepared.stopCapacity;
      const zoneCount = prepared.zones.featureCount;
      const tracks = session.output('stop-tracks', 'uint32', capacity);
      const startRows = session.output('start-rows', 'uint32', capacity);
      const endRows = session.output('end-rows', 'uint32', capacity);
      const centroids = session.output('centroids', 'float32x2', capacity);
      const durations = session.output('durations', 'float32', capacity);
      const stopZones = session.output('stop-zones', 'uint32', capacity);
      const count = session.output('count', 'uint32', 1);
      const requiredCount = session.output('required-count', 'uint32', 1);
      const stopOverflow = session.output('stop-overflow', 'uint32', 1);
      const joinOverflow = session.output('join-overflow', 'uint32', 1);
      const joinUncertainCount = session.output('join-uncertain', 'uint32', 1);
      const tableOverflow = session.output('table-overflow', 'uint32', 1);
      const tableCount = session.output('table-count', 'uint32', 1);
      const counts = session.output('zone-counts', 'uint32', zoneCount);
      const totals = session.output('zone-totals', 'float32', zoneCount);
      const means = session.output('zone-means', 'float32', zoneCount);
      const maximums = session.output('zone-maximums', 'float32', zoneCount);
      const recipe = addFleetDwellRecipe(session.graph, {
        id: BINDING_ID,
        spatialContext: prepared.trajectory.frame.gpuFrame?.spatialContext,
        trajectories: {
          kind: 'trajectories',
          positions: session.input('positions', prepared.trajectory.positions, 'float32x2'),
          timestamps: session.input('timestamps', prepared.trajectory.timestamps, 'uint32x2'),
          trackOffsets: session.input('track-offsets', prepared.trajectory.trackOffsets, 'uint32')
        },
        parameters: session.input('parameters', prepared.metricParameters, 'float32'),
        stopCapacity: capacity,
        zones: {
          kind: 'polygons',
          positions: session.input('zone-positions', prepared.zones.positions, 'float32x2'),
          featureOffsets: session.input('zone-features', prepared.zones.featureOffsets, 'uint32'),
          polygonOffsets: session.input('zone-polygons', prepared.zones.polygonOffsets, 'uint32'),
          ringOffsets: session.input('zone-rings', prepared.zones.ringOffsets, 'uint32'),
          candidateCapacity: prepared.candidateCapacity
        },
        zoneCapacity: zoneCount,
        outputs: {
          stops: {
            ids: tracks.view,
            startRows: startRows.view,
            endRows: endRows.view,
            centroids: centroids.view,
            durations: durations.view,
            status: {
              count: count.view,
              overflow: stopOverflow.view,
              requiredCount: requiredCount.view
            }
          },
          table: {
            counts: counts.view,
            sumValues: totals.view,
            means: means.view,
            maximums: maximums.view,
            status: {count: tableCount.view, overflow: tableOverflow.view}
          },
          status: {joinOverflow: joinOverflow.view, joinUncertainCount: joinUncertainCount.view}
        },
        scratch: {stopZones: stopZones.view}
      });
      session.run();
      const result: FleetDwellReadback = {
        count: await session.readScalar(count),
        requiredCount: await session.readScalar(requiredCount),
        stopOverflow: await session.readScalar(stopOverflow),
        joinOverflow: await session.readScalar(joinOverflow),
        joinUncertainCount: await session.readScalar(joinUncertainCount),
        tableOverflow: await session.readScalar(tableOverflow),
        tableCount: await session.readScalar(tableCount),
        tracks: await session.readUint32(tracks),
        startRows: await session.readUint32(startRows),
        endRows: await session.readUint32(endRows),
        centroids: await session.readFloat32(centroids),
        durations: await session.readFloat32(durations),
        stopZones: await session.readUint32(stopZones),
        counts: await session.readUint32(counts),
        totals: await session.readFloat32(totals),
        means: await session.readFloat32(means),
        maximums: await session.readFloat32(maximums)
      };
      return {
        ...makeFleetDwellOutput(prepared, result),
        dispatchedContributors: getFleetDwellContributorNames(recipe.contributors)
      };
    });
  }
};
