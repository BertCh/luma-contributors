// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {
  GPUPointCloudICP,
  GPUPointCloudChange,
  GPUPointCloudVoxelMerge
} from '@luma-contributors/analysis/pointcloud-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {addPointCloudFeatures, makePointCloudDiagnostics} from './catalog-pointcloud-neighborhoods';
import {
  makePointCloudTable,
  type PointCloudParameters,
  type PreparedPointCloudInputs
} from './catalog-pointcloud-input';

export async function runPointCloudRegistration(
  context: CatalogGpuContext,
  operation: string,
  prepared: PreparedPointCloudInputs,
  parameters: PointCloudParameters
): Promise<CatalogRunOutput> {
  if (operation === 'voxel-merge') return runVoxelMerge(context, prepared, parameters);
  if (operation === 'change')
    validateRequirement(
      'catalog-pointcloud-change',
      parameters['alreadyAligned'] === true,
      'alreadyAligned must certify a shared aligned frame'
    );
  const cloud = prepared.clouds[operation === 'icp' ? 'source' : 'reference'];
  const other = prepared.clouds[operation === 'icp' ? 'target' : 'comparison'];
  const count = cloud.collection.features.length;
  return withCatalogGpuSession(
    context.device,
    `catalog-pointcloud-${operation}`,
    async (session): Promise<CatalogRunOutput> => {
      if (operation === 'icp') {
        const iterations = Number(parameters['iterations']);
        const transform = session.output('transform', 'float32', 16);
        const history = session.output('history', 'float32x3', iterations);
        const inlierMask = session.output('inlier-mask', 'uint32', count);
        const status = session.output('status', 'uint32', 1);
        session.graph.add(
          new GPUPointCloudICP({
            source: session.input('source', cloud.positions, 'float32x3'),
            target: session.input('target', other.positions, 'float32x3'),
            maximumDistance: Number(parameters['maximumDistance']),
            iterations,
            transform: transform.view,
            history: history.view,
            inlierMask: inlierMask.view,
            status: status.view
          })
        );
        session.run();
        const matrix = await session.readFloat32(transform);
        const statusValue = await session.readScalar(status);
        const historyValues = await session.readFloat32(history);
        const transformedCoordinates = cloud.coordinates.map((_, row) => {
          const position = cloud.positions.subarray(row * 3, row * 3 + 3);
          const transformed = [0, 1, 2].map(
            axis =>
              matrix[axis] * position[0] +
              matrix[4 + axis] * position[1] +
              matrix[8 + axis] * position[2] +
              matrix[12 + axis] +
              prepared.origin[axis]
          );
          return [...prepared.frame.inverse(transformed[0], transformed[1]), transformed[2]];
        });
        const columns = {
          inlier: Array.from(await session.readUint32(inlierMask)),
          registration_status: Array<number>(count).fill(statusValue),
          original_longitude: cloud.coordinates.map(position => position[0]),
          original_latitude: cloud.coordinates.map(position => position[1]),
          original_altitude: cloud.coordinates.map(position => position[2])
        };
        const registrationColumns = {
          iteration: Array.from({length: iterations}, (_, index) => index),
          matched_count: Array.from({length: iterations}, (_, index) => historyValues[index * 3]),
          root_mean_square_error: Array.from(
            {length: iterations},
            (_, index) => historyValues[index * 3 + 1]
          ),
          translation_step: Array.from(
            {length: iterations},
            (_, index) => historyValues[index * 3 + 2]
          ),
          registration_status: Array<number>(iterations).fill(statusValue),
          ...Object.fromEntries(
            Array.from(matrix, (value, component) => [
              `transform_${component}`,
              Array<number>(iterations).fill(value)
            ])
          )
        };
        return {
          tables: [
            makePointCloudTable(
              {...cloud, coordinates: transformedCoordinates},
              operation,
              columns
            ),
            {
              id: 'registration',
              label: 'Registration transform and history',
              rowCount: iterations,
              columns: registrationColumns,
              columnTypes: Object.fromEntries(
                Object.keys(registrationColumns).map(name => [name, 'float64' as const])
              )
            }
          ],
          diagnostics: makePointCloudDiagnostics(columns),
          summary: {
            pointCount: count,
            registrationStatus: statusValue,
            originX: prepared.origin[0],
            originY: prepared.origin[1],
            originZ: prepared.origin[2],
            transformFrame: 'Shared local metric frame; column-major source-to-target'
          },
          dispatchedContributors: ['GPUPointCloudICP']
        };
      }
      const features = addPointCloudFeatures(session, cloud, parameters);
      const changes = session.output('changes', 'float32x3', count);
      const roughness = session.output('roughness', 'float32x3', count);
      const counts = session.output('epoch-counts', 'uint32', count * 2);
      const status = session.output('change-status', 'uint32', count);
      session.graph.add(
        new GPUPointCloudChange({
          reference: features.neighborhoods.positions,
          comparison: session.input('comparison', other.positions, 'float32x3'),
          cores: features.neighborhoods.positions,
          normals: features.normals.view,
          alignment: 'already-aligned',
          radius: Number(parameters['radius']),
          halfDepth: Number(parameters['halfDepth']),
          minimumSupport: Number(parameters['minimumSupport']),
          registrationError: Number(parameters['registrationError']),
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        })
      );
      session.run();
      const flags = await session.readUint32(status);
      const featureFlags = await session.readUint32(features.status);
      const epochCounts = await session.readUint32(counts);
      const columns: Record<string, (number | null)[]> = {
        change_status: Array.from(flags),
        feature_status: Array.from(featureFlags),
        neighborhood_status: Array.from(await session.readUint32(features.neighborhoods.status)),
        reference_count: Array.from({length: count}, (_, row) => epochCounts[row * 2]),
        comparison_count: Array.from({length: count}, (_, row) => epochCounts[row * 2 + 1])
      };
      for (const [names, output] of [
        [['change', 'lower_bound', 'upper_bound'], changes],
        [['reference_roughness', 'comparison_roughness', 'detection_limit'], roughness],
        [['normal_x', 'normal_y', 'normal_z'], features.normals]
      ] as const) {
        const values = await session.readFloat32(output);
        names.forEach((name, component) => {
          columns[name] = Array.from(flags, (flag, row) =>
            flag !== 0 || featureFlags[row] !== 0 || !Number.isFinite(values[row * 3 + component])
              ? null
              : values[row * 3 + component]
          );
        });
      }
      return {
        tables: [makePointCloudTable(cloud, operation, columns)],
        diagnostics: makePointCloudDiagnostics(columns),
        summary: {
          pointCount: count,
          comparisonCount: other.collection.features.length,
          alignment: 'already-aligned',
          originX: prepared.origin[0],
          originY: prepared.origin[1],
          originZ: prepared.origin[2]
        },
        dispatchedContributors: [
          'GPUPointCloudNeighborhoods',
          'GPUPointCloudFeatures',
          'GPUPointCloudChange'
        ]
      };
    }
  );
}

async function runVoxelMerge(
  context: CatalogGpuContext,
  prepared: PreparedPointCloudInputs,
  parameters: PointCloudParameters
): Promise<CatalogRunOutput> {
  const identifier = 'catalog-pointcloud-voxel-merge';
  const left = prepared.clouds['left'];
  const right = prepared.clouds['right'];
  const readProperty = (name: string, property: string) =>
    prepared.clouds[name].collection.features.map(feature => feature.properties?.[property]);
  const voxelProperty = String(parameters['voxelProperty']);
  const leftKeys = readProperty('left', voxelProperty);
  const rightKeys = readProperty('right', voxelProperty);
  validateRequirement(
    identifier,
    leftKeys.length === rightKeys.length &&
      leftKeys.every(
        (key, index) =>
          (typeof key === 'string' || (typeof key === 'number' && Number.isFinite(key))) &&
          typeof key === typeof leftKeys[0] &&
          key === rightKeys[index]
      ) &&
      new Set(leftKeys).size === leftKeys.length,
    'voxel IDs must be unique and identical in both parts and in the same order'
  );
  const parts = ['left', 'right'].map(name => {
    const counts = prepared.clouds[name].voxelCounts!;
    const representatives = readProperty(name, String(parameters['representativeProperty']));
    validateRequirement(
      identifier,
      representatives.every(
        (value, index) =>
          typeof value === 'number' &&
          Number.isInteger(value) &&
          value >= 0 &&
          value <= (counts[index] === 0 ? 0xffffffff : 0xfffffffe)
      ),
      `${name}.representatives must be valid uint32 source rows`
    );
    return {
      cloud: prepared.clouds[name],
      counts,
      representatives: Uint32Array.from(representatives as number[]),
      sourceRowOffset: name === 'right' ? Number(parameters['rightRowOffset']) : 0
    };
  });
  const count = left.collection.features.length;
  return withCatalogGpuSession(
    context.device,
    identifier,
    async (session): Promise<CatalogRunOutput> => {
      const counts = session.output('counts', 'uint32', count);
      const centroids = session.output('centroids', 'float32x3', count);
      const representatives = session.output('representatives', 'uint32', count);
      const status = session.output('status', 'uint32', count);
      session.graph.add(
        new GPUPointCloudVoxelMerge({
          partitionPolicy: 'disjoint-owned-rows',
          parts: parts.map((part, index) => ({
            counts: session.input(`counts-${index}`, part.counts, 'uint32'),
            centroids: session.input(`centroids-${index}`, part.cloud.positions, 'float32x3'),
            representatives: session.input(
              `representatives-${index}`,
              part.representatives,
              'uint32'
            ),
            sourceRowOffset: part.sourceRowOffset
          })),
          counts: counts.view,
          centroids: centroids.view,
          representatives: representatives.view,
          status: status.view
        })
      );
      session.run();
      const countValues = await session.readUint32(counts);
      const centroidValues = await session.readFloat32(centroids);
      const flags = await session.readUint32(status);
      const representativeValues = await session.readUint32(representatives);
      const coordinates = left.coordinates.map((_, row) =>
        [0, 1, 2].map(axis => centroidValues[row * 3 + axis] + prepared.origin[axis])
      );
      const table = makePointCloudTable({...left, coordinates}, 'voxel-merge', {
        voxel_id: leftKeys as (number | string)[],
        count: Array.from(countValues),
        representative: Array.from(representativeValues, (value, row) =>
          flags[row] || countValues[row] === 0 ? null : value
        ),
        merge_status: Array.from(flags),
        right_source_row: Array.from({length: count}, (_, row) => row),
        right_source_id: right.collection.features.map(feature =>
          feature.id == null ? null : String(feature.id)
        )
      });
      table.columnTypes['voxel_id'] = typeof leftKeys[0] === 'string' ? 'string' : 'float64';
      table.columnTypes['right_source_id'] = 'string';
      for (const name of ['longitude', 'latitude', 'altitude'])
        table.columns[name] = table.columns[name].map((value, row) =>
          flags[row] || countValues[row] === 0 ? null : value
        );
      return {
        tables: [table],
        diagnostics: makePointCloudDiagnostics(table.columns),
        summary: {voxelCount: count, partitionPolicy: 'disjoint-owned-rows'},
        dispatchedContributors: ['GPUPointCloudVoxelMerge']
      };
    }
  );
}
