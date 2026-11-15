// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUPointCloudNormalScale,
  GPUPointCloudStatisticalFilter,
  getGPUPointCloudNormalScaleParameterValues
} from '@luma-contributors/analysis/pointcloud-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import type {CatalogFeatureCollection} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  addPointCloudFeatures,
  addPointCloudNeighborhoods,
  makePointCloudDiagnostics
} from './catalog-pointcloud-neighborhoods';
import {
  makePointCloudTable,
  preparePointCloudInputs,
  validatePointCloudRadii,
  type PointCloudParameters
} from './catalog-pointcloud-input';
import {readPointCloudFeatureValue} from './catalog-pointcloud-feature-columns';
import {runPointCloudRaster} from './catalog-pointcloud-raster';
import {runPointCloudRegistration} from './catalog-pointcloud-registration';

export async function runPointCloudOperation(
  context: CatalogGpuContext,
  operation: string,
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  parameters: PointCloudParameters
): Promise<CatalogRunOutput> {
  const prepared = preparePointCloudInputs(inputs, parameters);
  if (['ground', 'smrf', 'surfaces', 'height'].includes(operation))
    return runPointCloudRaster(context, operation, prepared, parameters);
  if (['icp', 'change', 'voxel-merge'].includes(operation))
    return runPointCloudRegistration(context, operation, prepared, parameters);
  const cloud = prepared.clouds['points'];
  return withCatalogGpuSession(context.device, `catalog-pointcloud-${operation}`, async session => {
    const count = cloud.collection.features.length;
    const columns: Record<string, (number | null)[]> = {};
    let contributors: string[];
    if (operation === 'normal-scale') {
      const radii = parameters['radii'] as readonly number[];
      validatePointCloudRadii(radii);
      const normals = session.output('normals', 'float32x3', count);
      const scales = session.output('normal-scales', 'float32', count);
      const variances = session.output('normal-variances', 'float32', count);
      const status = session.output('normal-status', 'uint32', count);
      session.graph.add(
        new GPUPointCloudNormalScale({
          positions: session.input('positions', cloud.positions, 'float32x3'),
          radii,
          capacity: Number(parameters['capacity']),
          parameters: session.input(
            'parameters',
            getGPUPointCloudNormalScaleParameterValues({
              minimumSupport: Number(parameters['minimumSupport'])
            }),
            'float32'
          ),
          normals: normals.view,
          normalScales: scales.view,
          normalVariances: variances.view,
          status: status.view
        })
      );
      session.run();
      const flags = await session.readUint32(status);
      const normalValues = await session.readFloat32(normals);
      columns['normal_status'] = Array.from(flags);
      columns['normal_scale'] = Array.from(await session.readFloat32(scales), (value, row) =>
        flags[row] & 1 ? null : value
      );
      columns['normal_variance'] = Array.from(await session.readFloat32(variances), (value, row) =>
        flags[row] & 1 ? null : value
      );
      ['x', 'y', 'z'].forEach((axis, component) => {
        columns[`normal_${axis}`] = Array.from(flags, (flag, row) =>
          flag & 1 ? null : normalValues[row * 3 + component]
        );
      });
      contributors = ['GPUPointCloudNormalScale'];
    } else {
      const features =
        operation === 'features' ? addPointCloudFeatures(session, cloud, parameters) : undefined;
      const neighborhoods =
        features?.neighborhoods ?? addPointCloudNeighborhoods(session, cloud, parameters);
      const meanDistances =
        operation === 'statistical-filter'
          ? session.output('mean-distances', 'float32', count)
          : undefined;
      const validity = meanDistances ? session.output('validity', 'uint32', count) : undefined;
      if (meanDistances && validity)
        session.graph.add(
          new GPUPointCloudStatisticalFilter({
            squaredDistances: neighborhoods.squaredDistances.view,
            counts: neighborhoods.counts.view,
            capacity: neighborhoods.capacity,
            standardDeviations: Number(parameters['standardDeviations']),
            meanDistances: meanDistances.view,
            validity: validity.view
          })
        );
      session.run();
      const neighborCounts = await session.readUint32(neighborhoods.counts);
      columns['neighbor_count'] = Array.from(neighborCounts);
      columns['neighborhood_status'] = Array.from(await session.readUint32(neighborhoods.status));
      contributors = ['GPUPointCloudNeighborhoods'];
      if (features) {
        const flags = await session.readUint32(features.status);
        columns['feature_status'] = Array.from(flags);
        for (const [names, output, kind] of [
          [['normal_x', 'normal_y', 'normal_z'], features.normals, 'normal'],
          [['eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3'], features.eigenvalues, 'eigenvalue'],
          [['linearity', 'planarity', 'scattering'], features.shapes, 'shape']
        ] as const) {
          const values = await session.readFloat32(output);
          names.forEach((name, component) => {
            columns[name] = Array.from(flags, (flag, row) =>
              readPointCloudFeatureValue(
                values[row * 3 + component],
                kind,
                flag,
                Math.min(neighborCounts[row], neighborhoods.capacity)
              )
            );
          });
        }
        const covariance = await session.readFloat32(features.covariance);
        ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'].forEach((name, component) => {
          columns[`covariance_${name}`] = Array.from(flags, (flag, row) =>
            readPointCloudFeatureValue(
              covariance[row * 6 + component],
              'covariance',
              flag,
              Math.min(neighborCounts[row], neighborhoods.capacity)
            )
          );
        });
        contributors.push('GPUPointCloudFeatures');
      } else if (meanDistances && validity) {
        columns['mean_distance'] = Array.from(await session.readFloat32(meanDistances), value =>
          value < 0 ? null : value
        );
        columns['inlier'] = Array.from(await session.readUint32(validity));
        contributors.push('GPUPointCloudStatisticalFilter');
      } else {
        const indices = await session.readUint32(neighborhoods.indices);
        const distances = await session.readFloat32(neighborhoods.squaredDistances);
        for (let slot = 0; slot < neighborhoods.capacity; slot++) {
          columns[`neighbor_${slot + 1}_row`] = Array.from({length: count}, (_, row) =>
            indices[row * neighborhoods.capacity + slot] === 0xffffffff
              ? null
              : indices[row * neighborhoods.capacity + slot]
          );
          columns[`neighbor_${slot + 1}_distance`] = Array.from({length: count}, (_, row) =>
            indices[row * neighborhoods.capacity + slot] === 0xffffffff
              ? null
              : Math.sqrt(distances[row * neighborhoods.capacity + slot])
          );
        }
      }
    }
    return {
      tables: [makePointCloudTable(cloud, operation, columns)],
      diagnostics: makePointCloudDiagnostics(columns),
      summary: {
        pointCount: count,
        originX: prepared.origin[0],
        originY: prepared.origin[1],
        originZ: prepared.origin[2]
      },
      dispatchedContributors: contributors
    };
  });
}
