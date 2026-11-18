// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Dimensions and optional output selection shared by geometric-median execution forms. */
export type GPURasterGeometricMedianCompositeOutputLayoutOptions = {
  /** ID used in validation messages. */
  id?: string;
  /** Number of raster cells, a positive uint32 count. */
  cellCount: number;
  /** Components per composite, a positive uint32 count. GPU execution supports at most 16. */
  bandCount: number;
  /** Include the per-cell final-step length. Defaults to false. */
  lastStep?: boolean;
};

/** Scalar storage requirements; an omitted lastStep has no output descriptor. */
export type GPURasterGeometricMedianCompositeOutputLayout = {
  composite: {format: 'float32'; length: number};
  status: {format: 'uint32'; length: number};
  lastStep?: {format: 'float32'; length: number};
};

/**
 * Describes dense cell-major output without allocating storage. composite[cell * bandCount + band]
 * has cellCount * bandCount entries; status and requested lastStep have one entry per cell.
 * Counts and products must fit uint32 indexing. GPU slice, band and iteration limits remain with
 * the contributor; the layout also describes the CPU twin, which permits larger band counts.
 * Caller-owned packed outputs may have spare capacity; this describes only their used prefix.
 */
export function getGPURasterGeometricMedianCompositeOutputLayout(
  options: GPURasterGeometricMedianCompositeOutputLayoutOptions
): GPURasterGeometricMedianCompositeOutputLayout {
  const {id = 'getGPURasterGeometricMedianCompositeOutputLayout', cellCount, bandCount} = options;
  validateCapacity(id, 'cellCount', cellCount);
  validateCapacity(id, 'bandCount', bandCount);
  const compositeLength = cellCount * bandCount;
  validateCapacity(id, 'cellCount times bandCount', compositeLength);
  return {
    composite: {format: 'float32', length: compositeLength},
    status: {format: 'uint32', length: cellCount},
    ...(options.lastStep ? {lastStep: {format: 'float32' as const, length: cellCount}} : {})
  };
}
