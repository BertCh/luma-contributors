// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Private test utilities shared by this workspace's analysis and lenses tests. Resolved through
 * the `paths` alias in tsconfig.json; not a published export of `@luma-contributors/analysis`.
 * Import only from test code: it depends on `vitest` and `wgsl_reflect`.
 */
export * from './gpu-contributor-test-utils';
export {expectNumericArrayClose} from './expect-numeric-array-close';
export type {NumericArrayTolerance} from './expect-numeric-array-close';
export {requireTestDevice} from './require-test-device';

// Internals that dependent packages' tests exercise directly.
export {importGraphBuffer} from '../analysis-core/kernels/gpu-contributor-utils';
export {
  getGPUResourceCounts,
  getGPUResourceLeaks,
  validateNoGPUResourceLeaks
} from './gpu-resource-counts';
export type {GPUResourceCounts} from './gpu-resource-counts';
