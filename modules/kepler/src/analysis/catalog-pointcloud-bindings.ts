// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  COORDINATE_SYSTEM_PARAMETER,
  type CatalogOperationBinding,
  type CatalogParameterSpec
} from './catalog-binding-types';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {runPointCloudOperation} from './catalog-pointcloud-gpu';
import {MAXIMUM_POINT_CLOUD_ROWS} from './catalog-pointcloud-input';
import {
  POINT_CLOUD_ICP_MAXIMUM_ITERATIONS_PARAMETER,
  REMOVED_POINT_CLOUD_ICP_ITERATIONS_MESSAGE
} from './catalog-pointcloud-icp-parameters';

const numberParameter = (
  name: string,
  label: string,
  defaultValue: number,
  minimum: number,
  maximum: number
): CatalogParameterSpec => ({
  name,
  label,
  kind: 'number',
  defaultValue,
  minimum,
  maximum,
  unit: 'crs'
});
const integerParameter = (
  name: string,
  label: string,
  defaultValue: number,
  minimum: number,
  maximum: number
): CatalogParameterSpec => ({name, label, kind: 'integer', defaultValue, minimum, maximum});
const propertyParameter = (
  name: string,
  label: string,
  defaultValue: string
): CatalogParameterSpec => ({name, label, kind: 'string', defaultValue, maximumLength: 128});
const radius = numberParameter('radius', 'Neighborhood radius', 3, 0.000001, 1000000);
const capacity = integerParameter('capacity', 'Retained neighbors', 32, 1, 64);
const cellSize = numberParameter('cellSize', 'Raster cell size', 1, 0.000001, 1000000);
const classification = propertyParameter(
  'classificationProperty',
  'LAS classification property',
  'classification'
);
const radii: CatalogParameterSpec = {
  name: 'radii',
  label: 'Increasing radii',
  unit: 'crs',
  kind: 'number-list',
  minimum: 0.000001,
  maximum: 32,
  maximumLength: 8,
  defaultValue: [1, 2]
};
const definitions: {
  id: string;
  title: string;
  contributors: string[];
  capability: string;
  parameters: CatalogParameterSpec[];
  inputs?: string[];
  limitation?: string;
}[] = [
  {
    id: 'ground',
    title: 'Progressive morphological ground',
    contributors: ['GPUPointCloudSurfaces', 'GPUPointCloudGround'],
    capability: 'pointcloud-ground',
    parameters: [
      cellSize,
      classification,
      {...radii, label: 'Morphology radii', unit: 'cells'},
      numberParameter('initialThreshold', 'Initial elevation threshold', 0.2, 0, 10000),
      numberParameter('slope', 'Terrain slope', 0.1, 0, 100),
      numberParameter('maximumThreshold', 'Maximum elevation threshold', 2, 0, 10000)
    ],
    limitation:
      'A surface pass first computes unclassified cell minima from all points, independent of LAS class. Square-window radius-increment thresholds follow the contributor; they are not the Zhang full-window-width convention. Radii must be integers.'
  },
  {
    id: 'smrf',
    title: 'SMRF ground classification',
    contributors: ['GPUPointCloudSMRF'],
    capability: 'pointcloud-ground',
    parameters: [
      cellSize,
      classification,
      numberParameter('slope', 'Terrain slope', 0.15, 0, 100),
      numberParameter('threshold', 'Elevation threshold', 0.5, 0, 10000),
      numberParameter('scalar', 'Slope scalar', 1.25, 0, 100),
      integerParameter('windowCells', 'Window radius in cells', 3, 1, 8)
    ]
  },
  {
    id: 'surfaces',
    title: 'Terrain, surface and canopy',
    contributors: ['GPUPointCloudSurfaces'],
    capability: 'pointcloud-ground',
    parameters: [cellSize, classification]
  },
  {
    id: 'height',
    title: 'Height above ground',
    contributors: ['GPUPointCloudSurfaces', 'GPUPointCloudHeight'],
    capability: 'pointcloud-ground',
    parameters: [
      cellSize,
      classification,
      {
        name: 'interpolation',
        label: 'Terrain interpolation',
        kind: 'enum',
        options: ['cell', 'bilinear'],
        defaultValue: 'cell'
      }
    ]
  },
  {
    id: 'neighborhoods',
    title: '3D point neighborhoods',
    contributors: ['GPUPointCloudNeighborhoods'],
    capability: 'pointcloud-neighborhoods',
    parameters: [radius, capacity]
  },
  {
    id: 'features',
    title: 'Point normals and covariance features',
    contributors: ['GPUPointCloudNeighborhoods', 'GPUPointCloudFeatures'],
    capability: 'pointcloud-neighborhoods',
    parameters: [radius, capacity]
  },
  {
    id: 'statistical-filter',
    title: 'Statistical point outliers',
    contributors: ['GPUPointCloudNeighborhoods', 'GPUPointCloudStatisticalFilter'],
    capability: 'pointcloud-neighborhoods',
    parameters: [
      radius,
      capacity,
      numberParameter('standardDeviations', 'Standard deviations', 2, 0, 100)
    ]
  },
  {
    id: 'normal-scale',
    title: 'Multiscale point normals',
    contributors: ['GPUPointCloudNormalScale'],
    capability: 'pointcloud-registration',
    parameters: [radii, capacity, integerParameter('minimumSupport', 'Minimum support', 3, 3, 64)]
  },
  {
    id: 'icp',
    title: 'Rigid point-cloud registration',
    contributors: ['GPUPointCloudICP'],
    capability: 'pointcloud-registration',
    inputs: ['source', 'target'],
    parameters: [
      numberParameter('maximumDistance', 'Maximum correspondence distance', 3, 0.000001, 1000000),
      POINT_CLOUD_ICP_MAXIMUM_ITERATIONS_PARAMETER
    ],
    limitation:
      'Point-to-point ICP with identity initialization. Transform is source to target in the shared local metric frame. Inspect registration status before using an unconverged alignment.'
  },
  {
    id: 'change',
    title: 'Normal-direction cloud change',
    contributors: ['GPUPointCloudNeighborhoods', 'GPUPointCloudFeatures', 'GPUPointCloudChange'],
    capability: 'pointcloud-registration',
    inputs: ['reference', 'comparison'],
    parameters: [
      radius,
      capacity,
      numberParameter('halfDepth', 'Cylinder half depth', 2, 0.000001, 1000000),
      integerParameter('minimumSupport', 'Minimum epoch support', 3, 2, 2048),
      numberParameter('registrationError', 'Registration uncertainty', 0, 0, 10000),
      {
        name: 'alreadyAligned',
        label: 'Epochs share an aligned frame',
        kind: 'boolean',
        defaultValue: false
      }
    ],
    limitation:
      'Explicit aligned-frame certification is required. Reference rows are cores; normals use positive-z orientation. No registration or datum correction is inferred.'
  },
  {
    id: 'voxel-merge',
    title: 'Merge voxel sufficient statistics',
    contributors: ['GPUPointCloudVoxelMerge'],
    capability: 'pointcloud-merge',
    inputs: ['left', 'right'],
    parameters: [
      propertyParameter('voxelProperty', 'Voxel ID property', 'voxel_id'),
      propertyParameter('countProperty', 'Voxel count property', 'count'),
      propertyParameter('representativeProperty', 'Representative row property', 'representative'),
      integerParameter('rightRowOffset', 'Right source row offset', 0, 0, 2147483647),
      {
        name: 'disjointOwnedRows',
        label: 'Parts own disjoint rows on the same voxel grid',
        kind: 'boolean',
        defaultValue: false
      }
    ],
    limitation:
      'Inputs are per-voxel arithmetic centroids, counts and representatives on one identical grid, not raw points or downsample representatives. Both inputs must list identical unique voxel IDs in identical order. Halo replicas must be excluded by the caller.'
  }
];

/** All eleven numerical contributors exported by pointcloud-analysis, through bounded host adapters. */
export const POINT_CLOUD_BINDINGS: readonly CatalogOperationBinding[] = definitions.map(
  definition => ({
    id: `pointcloud-${definition.id}`,
    title: definition.title,
    domain: 'analysis',
    summary: `${definition.title} on explicit XYZ points; publish numeric columns for native point styling and altitude.`,
    capabilityIds: [definition.capability],
    contributorNames: definition.contributors,
    gpuContributorNames: definition.contributors,
    engine: 'gpu',
    inputs: (definition.inputs ?? ['points']).map(name => ({
      name,
      label: `${name[0].toUpperCase()}${name.slice(1)} cloud`,
      accepts: ['point'],
      maximumFeatures: MAXIMUM_POINT_CLOUD_ROWS,
      maximumVertices: MAXIMUM_POINT_CLOUD_ROWS
    })),
    parameters: [COORDINATE_SYSTEM_PARAMETER, ...definition.parameters],
    outputs: [
      {
        tableId: 'points',
        label: definition.title,
        geometry: 'point',
        identityColumns: ['result_id', 'source_row', 'source_id'],
        maximumRows: MAXIMUM_POINT_CLOUD_ROWS
      },
      ...(definition.id === 'surfaces'
        ? [
            {
              tableId: 'grid',
              label: 'Point-cloud surfaces',
              geometry: 'point' as const,
              identityColumns: ['cell_id'],
              maximumRows: 4096
            }
          ]
        : []),
      ...(definition.id === 'icp'
        ? [
            {
              tableId: 'registration',
              label: 'Registration transform and history',
              geometry: 'none' as const,
              identityColumns: ['iteration'],
              maximumRows: 30
            }
          ]
        : [])
    ],
    cleanup:
      'The catalog session releases compiled graphs and every input, scratch and output buffer on success or failure. Bounded readback becomes Arrow columns published through Sources.',
    declaredBehavior: [
      'At most 2,048 rows per cloud and 4,096 raster cells. Non-Point, missing XYZ, nonfinite coordinates and invalid classifications refuse the whole input; no row is silently dropped.',
      'EPSG:4326 uses the shared local azimuthal-equidistant frame in meters; altitude must be meters in the same vertical datum. Planar XYZ must share one Cartesian unit. Both epochs share one origin subtracted in binary64 before Float32 upload.',
      'Source rows, IDs, scalar properties and original altitude survive. Numeric status columns and warning diagnostics expose unsupported, truncated, degenerate and unconverged results. Invalid measurements publish null.',
      'Neighborhoods retain the nearest 1–64 other rows inside the inclusive 3D radius; counts preserve actual support. Statistical filtering marks rows without compacting or deleting them.',
      'Raster row zero is south/top; grid origin is the cloud XY minimum. More than 4,096 cells requires a larger cell size. Terrain for surfaces/height requires explicit LAS class 2 support; ground classification prepares minima over all classes. Unclassified inputs default to class 1.'
    ],
    limitation:
      definition.limitation ??
      'Bounded catalog readback and native Sources publication; no streaming, implicit unit/datum conversion, CPU numerical fallback or new numerical implementation. Raster dimensions are at least two cells.',
    runGpu: (context, inputs, parameters) => {
      if (definition.id === 'icp') {
        validateRequirement(
          'pointcloud-icp:',
          !('iterations' in parameters),
          REMOVED_POINT_CLOUD_ICP_ITERATIONS_MESSAGE
        );
      }
      return runPointCloudOperation(context, definition.id, inputs, parameters);
    }
  })
);
