// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the point-cloud cases. */
export const POINTCLOUD_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUPointCloudNormalScale',
    name: 'scale count one above 32',
    kind: 'dispatch',
    limit: 32,
    rejects: /^Error: gpu-pointcloud-normal-scale radii\.length must be an integer in \[1, 32\]$/
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'unindexed scan one core above the work limit',
    kind: 'dispatch',
    limit: 134217728,
    rejects:
      /^Error: gpu-pointcloud-change cores x \(reference \+ comparison\) is \d+, above the 134217728 limit of a scan without an index; pass grid$/
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'sampling covariance one point above 32768 per cloud',
    kind: 'dispatch',
    limit: 32768,
    rejects:
      /^Error: gpu-pointcloud-change samplingCovariance points per cloud must be at most 32768$/
  },
  {
    exportName: 'GPUPointCloudNeighborhoods',
    name: 'capacity one above 64',
    kind: 'binding',
    limit: 64,
    rejects: /^Error: gpu-pointcloud-neighborhoods capacity must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPUPointCloudNeighborhoods',
    name: 'unindexed search one query above the work limit',
    kind: 'dispatch',
    limit: 134217728,
    rejects:
      /^Error: gpu-pointcloud-neighborhoods queries x positions is \d+, above the 134217728 limit of a search without an index; pass grid$/
  },
  {
    exportName: 'GPUPointCloudNeighborhoods',
    name: 'grid axis one above 16777216 cells',
    kind: 'dispatch',
    limit: 16777216,
    rejects:
      /^Error: gpu-pointcloud-neighborhoods grid\.dimensions\[0\] must be an integer in \[1, 16777216\]$/
  },
  {
    exportName: 'GPUPointCloudFeatures',
    name: 'capacity one above 64',
    kind: 'binding',
    limit: 64,
    rejects: /^Error: gpu-pointcloud-features capacity must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPUPointCloudStatisticalFilter',
    name: 'capacity one above 64',
    kind: 'binding',
    limit: 64,
    rejects: /^Error: gpu-pointcloud-statistical-filter capacity must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPUPointCloudGround',
    name: 'thirty-three radii',
    kind: 'dispatch',
    limit: 32,
    rejects: /^Error: gpu-pointcloud-ground radii\.length must be an integer in \[1, 32\]$/
  },
  {
    exportName: 'GPUPointCloudGround',
    name: 'radius one above 32 cells',
    kind: 'dispatch',
    limit: 32,
    rejects: /^Error: gpu-pointcloud-ground radii\[0\] must be an integer in \[1, 32\]$/
  },
  {
    exportName: 'GPUPointCloudSMRF',
    name: 'window one above 64 cells',
    kind: 'dispatch',
    limit: 64,
    rejects: /^Error: gpu-pointcloud-smrf window in cells must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPUPointCloudSMRF',
    name: 'cut one above 32 cells',
    kind: 'dispatch',
    limit: 32,
    rejects: /^Error: gpu-pointcloud-smrf cut in cells must be an integer in \[0, 32\]$/
  },
  {
    exportName: 'GPUPointCloudICP',
    name: 'maximumIterations one above 100',
    kind: 'dispatch',
    limit: 100,
    rejects: /^Error: gpu-pointcloud-icp maximumIterations must be an integer in \[1, 100\]$/
  },
  {
    exportName: 'GPUPointCloudICP',
    name: 'unindexed search one source point above the work limit',
    kind: 'dispatch',
    limit: 134217728,
    rejects:
      /^Error: gpu-pointcloud-icp source x target is \d+, above the 134217728 limit of a search without an index; pass targetGrid$/
  },
  {
    exportName: 'GPUPointCloudHeight',
    name: 'inverse-distance power one above 8',
    kind: 'dispatch',
    limit: 8,
    rejects:
      /^Error: gpu-pointcloud-height interpolation\.power must be positive in float32 and at most GPU_POINT_CLOUD_MAXIMUM_IDW_POWER \(8\)$/
  },
  {
    exportName: 'GPUPointCloudChange',
    name: 'indexed cylinder visiting 102 cells per axis',
    kind: 'dispatch',
    limit: 1048576,
    rejects:
      /^Error: gpu-pointcloud-change each core would visit 1061208 grid cells, above the 1048576 limit; enlarge the grid cellSize$/
  },
  {
    exportName: 'GPUPointCloudNeighborhoods',
    name: 'grid of more than 4294967294 cells',
    kind: 'dispatch',
    limit: 4294967294,
    rejects: /^Error: gpu-pointcloud-neighborhoods grid cell count must be at most 4294967294$/
  },
  {
    exportName: 'GPUPointCloudHeight',
    name: 'inverse-distance reach one above 64 cells',
    kind: 'dispatch',
    limit: 64,
    rejects: /^Error: gpu-pointcloud-height interpolation\.radius in cells must be at most 64$/
  }
];
