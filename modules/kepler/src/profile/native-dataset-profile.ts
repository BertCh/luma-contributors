// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {countFeatureVertices, type CatalogFeature} from '../analysis/catalog-geojson';
import {
  MAXIMUM_TRAJECTORY_FIXES,
  prepareCatalogTrajectory
} from '../analysis/catalog-trajectory-common';
import {
  getKeplerDatasetProfileUndeclared,
  profileKeplerColumnTable,
  type KeplerColumnTableProfileOptions,
  type KeplerProfileColumnTable
} from './dataset-profile';
import type {KeplerDatasetProfile} from './dataset-profile-types';

// Match bounded host catalog readback. Native refinement must not expand an unbounded source.
const MAXIMUM_NATIVE_PROFILE_ROWS = 20000;

/** Explicit publication roles; no sample IDs or column-name guesses establish native readiness. */
export type KeplerNativeDatasetShape =
  | {kind: 'cells'}
  | {kind: 'scalar-grid'; width: number; height: number; valueColumn: string}
  | {
      kind: 'trajectories';
      trackColumn?: string;
      timeColumn?: string;
      timestampEncoding: 'unix-second' | 'unix-millisecond' | 'iso8601';
    };

/**
 * Refine prepared native sources using the existing profiler and catalog trajectory preparer.
 * Cell boundaries come from publication. Scalar grids declare continuous-surface samples without
 * inventing measurement levels. Every source row, including null geometry and nodata, is retained.
 */
export function profileKeplerNativeDataset(
  table: KeplerProfileColumnTable,
  options: KeplerColumnTableProfileOptions,
  shape: KeplerNativeDatasetShape
): KeplerDatasetProfile {
  const contributorId = 'profileKeplerNativeDataset';
  validateCapacity(
    contributorId,
    'rowCount',
    table.rowCount,
    0,
    shape.kind === 'trajectories' ? MAXIMUM_TRAJECTORY_FIXES : MAXIMUM_NATIVE_PROFILE_ROWS
  );
  if (shape.kind === 'scalar-grid') {
    validateCapacity(contributorId, 'width', shape.width, 1, MAXIMUM_NATIVE_PROFILE_ROWS);
    validateCapacity(contributorId, 'height', shape.height, 1, MAXIMUM_NATIVE_PROFILE_ROWS);
    validateRequirement(
      contributorId,
      table.rowCount === shape.width * shape.height,
      'rowCount must equal width * height for a dense scalar grid'
    );
    validateRequirement(
      contributorId,
      table.columns.some(column => column.name === shape.valueColumn),
      'valueColumn must name a scalar column'
    );
  }
  // Bound timed-path expansion before the general profiler allocates locations or computes spacing.
  if (shape.kind === 'trajectories' && options.geometryColumns?.length) {
    const geometryValues = table.columns.find(
      column => column.name === options.geometryColumns![0]
    )?.values;
    if (geometryValues) {
      let vertexCount = 0;
      for (let row = 0; row < geometryValues.length; row++) {
        const geometry = JSON.parse(String(geometryValues[row])) as CatalogFeature['geometry'];
        vertexCount += countFeatureVertices({geometry, properties: {}});
      }
      validateCapacity(contributorId, 'vertexCount', vertexCount, 0, MAXIMUM_TRAJECTORY_FIXES);
    }
  }
  const profile = profileKeplerColumnTable(table, {
    ...options,
    declarations: {
      ...(shape.kind === 'scalar-grid' ? {rowSemantics: 'surface-samples' as const} : {}),
      ...options.declarations
    }
  });
  const undeclared = getKeplerDatasetProfileUndeclared(profile);
  if (shape.kind === 'cells') {
    validateRequirement(
      contributorId,
      profile.catalogGeometry === 'polygon',
      'cells require prepared polygon boundaries'
    );
    profile.kind = 'cells';
    undeclared.kind = 'cells';
    return profile;
  }
  if (shape.kind === 'scalar-grid') {
    validateRequirement(
      contributorId,
      profile.catalogGeometry === 'point',
      'scalar-grid requires explicit cell-center coordinates'
    );
    validateRequirement(
      contributorId,
      profile.columns.some(
        column =>
          column.name === shape.valueColumn &&
          (column.type === 'number' || column.type === 'integer')
      ),
      'valueColumn must contain numeric scalars'
    );
    profile.kind = 'cells';
    profile.vertexCount = profile.spacing?.pointCount ?? 0;
    profile.scalarGrid = {width: shape.width, height: shape.height, valueColumn: shape.valueColumn};
    Object.assign(undeclared, {
      kind: 'cells',
      vertexCount: profile.vertexCount,
      scalarGrid: profile.scalarGrid,
      rowSemantics: {
        value: 'surface-samples',
        source: 'declared',
        reason: 'Declared continuous scalar grid'
      }
    });
    return profile;
  }
  validateRequirement(
    contributorId,
    ['unix-second', 'unix-millisecond', 'iso8601'].includes(shape.timestampEncoding),
    'timestampEncoding must be unix-second, unix-millisecond or iso8601'
  );
  validateRequirement(
    contributorId,
    profile.catalogGeometry === 'point' || profile.catalogGeometry === 'line',
    'trajectories require point fixes or timed lines'
  );
  validateRequirement(
    contributorId,
    profile.catalogGeometry !== 'point' || Boolean(shape.trackColumn && shape.timeColumn),
    'point trajectories require trackColumn and timeColumn'
  );
  for (const name of [shape.trackColumn, shape.timeColumn].filter(
    (name): name is string => name !== undefined
  )) {
    validateRequirement(
      contributorId,
      table.columns.some(column => column.name === name),
      `trajectory column ${name} must exist`
    );
  }
  const parameters = {
    coordinateSystem: profile.coordinates?.model === 'planar' ? 'planar' : 'EPSG:4326',
    trackProperty: shape.trackColumn ?? '',
    timeProperty: shape.timeColumn ?? '',
    timestampEncoding: shape.timestampEncoding,
    longitudeProperty: profile.pointColumns?.longitude ?? '',
    latitudeProperty: profile.pointColumns?.latitude ?? ''
  };
  const columns = new Map(table.columns.map(column => [column.name, column.values]));
  const geometryColumn = profile.geometryColumns[0];
  const features: CatalogFeature[] = Array.from({length: table.rowCount}, (_, row) => ({
    id: row,
    properties: Object.fromEntries(
      table.columns
        .filter(column => column.name !== geometryColumn)
        .map(column => [column.name, column.values[row]])
    ),
    geometry: geometryColumn
      ? (JSON.parse(String(columns.get(geometryColumn)![row])) as CatalogFeature['geometry'])
      : {
          type: 'Point',
          coordinates: [
            columns.get(parameters.longitudeProperty)![row] as number,
            columns.get(parameters.latitudeProperty)![row] as number
          ]
        }
  }));
  const trajectory = prepareCatalogTrajectory({type: 'FeatureCollection', features}, parameters);
  profile.kind = 'trajectories';
  profile.vertexCount = trajectory.fixes.length;
  profile.trajectory = {
    parameters,
    fixCount: trajectory.fixes.length,
    trackCount: trajectory.trackIds.length
  };
  Object.assign(undeclared, {
    kind: profile.kind,
    vertexCount: profile.vertexCount,
    trajectory: profile.trajectory
  });
  return profile;
}
