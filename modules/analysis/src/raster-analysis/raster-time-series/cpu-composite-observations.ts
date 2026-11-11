// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Compile-time packed stack dimensions shared by CPU compositors. @internal */
export type CPUCompositeDimensions = {cellCount: number; sliceCount: number; bandCount: number};

/** Validates shape products against uint32 indexing, without GPU per-invocation caps. @internal */
export function validateCPUCompositeDimensions(
  id: string,
  dimensions: CPUCompositeDimensions
): void {
  validateCapacity(id, 'cellCount', dimensions.cellCount);
  validateCapacity(id, 'sliceCount', dimensions.sliceCount);
  validateCapacity(id, 'bandCount', dimensions.bandCount);
  validateCapacity(id, 'cellCount times sliceCount', dimensions.cellCount * dimensions.sliceCount);
  validateCapacity(
    id,
    'cellCount times sliceCount times bandCount',
    dimensions.cellCount * dimensions.sliceCount * dimensions.bandCount
  );
}

/** Throws an argument-naming error for a short stack or mask column. @internal */
export function validateCPUCompositeArrays(
  values: Float32Array,
  mask: Uint32Array | undefined,
  dimensions: CPUCompositeDimensions
): void {
  const observationCount = dimensions.cellCount * dimensions.sliceCount;
  if (values.length < observationCount * dimensions.bandCount)
    throw new Error('values must hold cellCount times sliceCount times bandCount entries');
  if (mask && mask.length < observationCount)
    throw new Error('mask must hold cellCount times sliceCount entries');
}

/** Returns valid slice indices in source order, requiring all bands finite. @internal */
export function getCPUCompositeSlices(
  values: Float32Array,
  mask: Uint32Array | undefined,
  dimensions: CPUCompositeDimensions,
  cell: number
): number[] {
  const slices: number[] = [];
  for (let slice = 0; slice < dimensions.sliceCount; slice++) {
    const observation = cell * dimensions.sliceCount + slice;
    if ((mask?.[observation] ?? 1) === 0) continue;
    let valid = true;
    for (let band = 0; band < dimensions.bandCount; band++)
      if (!Number.isFinite(values[observation * dimensions.bandCount + band])) {
        valid = false;
        break;
      }
    if (valid) slices.push(slice);
  }
  return slices;
}
