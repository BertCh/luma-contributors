// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUPointCloudFeatures,
  GPUPointCloudNeighborhoods
} from '@luma-contributors/analysis/pointcloud-analysis';
import type {CatalogGpuSession} from './catalog-gpu-session';
import type {PointCloudParameters, PreparedPointCloud} from './catalog-pointcloud-input';

/** One graph for neighborhood search plus downstream features/filter; no intermediate readback. */
export function addPointCloudNeighborhoods(
  session: CatalogGpuSession,
  cloud: PreparedPointCloud,
  parameters: PointCloudParameters
) {
  const count = cloud.collection.features.length;
  const capacity = Number(parameters['capacity']);
  const positions = session.input('positions', cloud.positions, 'float32x3');
  const indices = session.output('indices', 'uint32', count * capacity);
  const squaredDistances = session.output('squared-distances', 'float32', count * capacity);
  const counts = session.output('counts', 'uint32', count);
  const status = session.output('neighborhood-status', 'uint32', count);
  session.graph.add(
    new GPUPointCloudNeighborhoods({
      positions,
      radius: Number(parameters['radius']),
      capacity,
      output: {
        indices: indices.view,
        squaredDistances: squaredDistances.view,
        counts: counts.view,
        status: status.view
      }
    })
  );
  return {positions, indices, squaredDistances, counts, status, capacity};
}

export function addPointCloudFeatures(
  session: CatalogGpuSession,
  cloud: PreparedPointCloud,
  parameters: PointCloudParameters
) {
  const neighborhoods = addPointCloudNeighborhoods(session, cloud, parameters);
  const count = cloud.collection.features.length;
  const normals = session.output('normals', 'float32x3', count);
  const eigenvalues = session.output('eigenvalues', 'float32x3', count);
  const shapes = session.output('shapes', 'float32x3', count);
  const covariance = session.output('covariance', 'float32', count * 6);
  const status = session.output('feature-status', 'uint32', count);
  session.graph.add(
    new GPUPointCloudFeatures({
      positions: neighborhoods.positions,
      indices: neighborhoods.indices.view,
      counts: neighborhoods.counts.view,
      capacity: neighborhoods.capacity,
      output: {
        normals: normals.view,
        eigenvalues: eigenvalues.view,
        shapes: shapes.view,
        covariance: covariance.view,
        status: status.view
      }
    })
  );
  return {neighborhoods, normals, eigenvalues, shapes, covariance, status};
}

export function makePointCloudDiagnostics(
  columns: Record<string, readonly (number | string | boolean | null)[]>
) {
  return Object.entries(columns)
    .filter(([name]) => name.endsWith('status'))
    .flatMap(([name, values]) => {
      const count = values.filter(value => typeof value === 'number' && value !== 0).length;
      return count
        ? [
            {
              code: name,
              severity: 'warning' as const,
              count,
              message: `${count} rows have nonzero ${name}; inspect the contributor status codes/bits before styling or interpreting measurements.`
            }
          ]
        : [];
    });
}
