// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the STX-04 and STX-05 cases. */
export const STX04_STX05_SPACE_TIME_DENSITY_AND_K_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUSpaceTimeKernelDensity',
    name: 'one slice above the voxel limit',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: space-time-kernel-density voxels \(grid\.width \* grid\.height \* sliceCount\) must be at most 33554432$/
  },
  {
    exportName: 'GPUSpaceTimeKernelDensity',
    name: 'one point above the row limit',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: space-time-kernel-density positions length must be at most 8388608$/
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'one row above the row limit',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: space-time-k-function positions length must be at most 8388608$/
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'local statistics one row above the binding limit',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: space-time-k-function output\.local rows must be at most 33554432$/
  },
  {
    exportName: 'GPUSpaceTimeDiggleTest',
    name: 'one permutation above the maximum',
    kind: 'dispatch',
    limit: 999,
    rejects: /^Error: space-time-diggle-test permutationCount must be an integer in \[1, 999\]$/
  }
];
