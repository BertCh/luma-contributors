// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_RASTER_MEDOID_NO_SLICE} from './gpu-raster-medoid-composite';
import type {RasterMedoidCompositeOperationProps} from './gpu-raster-medoid-composite-operation';
import {
  getCPUCompositeSlices,
  validateCPUCompositeArrays,
  validateCPUCompositeDimensions,
  type CPUCompositeDimensions
} from './cpu-composite-observations';

export type CPURasterMedoidCompositeResult = {
  composite: Float32Array;
  sliceIndex: Uint32Array;
  status: Uint32Array;
};

/**
 * Selects the real observation with minimum total multiband Euclidean distance to all valid
 * observations, with earliest-slice ties. A zero mask or any nonfinite band rejects the whole slice.
 *
 * What differs from the GPU contributor:
 * - Distances and sums use float64; near ties can select a different
 *   medoid. Output values are copied from the selected float32 observation.
 * - GPU limits of 64 slices and 16 bands are dropped; uint32 shape/indexing limits remain.
 *
 * @throws If values/mask are too short or dimensions are invalid.
 */
export function computeRasterMedoidCompositeOnCPU(
  values: Float32Array,
  dimensions: CPUCompositeDimensions,
  mask?: Uint32Array
): CPURasterMedoidCompositeResult {
  validateCPUCompositeDimensions('computeRasterMedoidCompositeOnCPU', dimensions);
  validateCPUCompositeArrays(values, mask, dimensions);
  const composite = new Float32Array(dimensions.cellCount * dimensions.bandCount).fill(NaN);
  const sliceIndex = new Uint32Array(dimensions.cellCount).fill(GPU_RASTER_MEDOID_NO_SLICE);
  const status = new Uint32Array(dimensions.cellCount).fill(1);
  for (let cell = 0; cell < dimensions.cellCount; cell++) {
    const slices = getCPUCompositeSlices(values, mask, dimensions, cell);
    let varyingBandCount = 0;
    let scalarBand = 0;
    if (slices.length > 0) {
      for (let band = 0; band < dimensions.bandCount; band++) {
        const firstValue =
          values[(cell * dimensions.sliceCount + slices[0]) * dimensions.bandCount + band];
        if (
          slices.some(
            slice =>
              values[(cell * dimensions.sliceCount + slice) * dimensions.bandCount + band] !==
              firstValue
          )
        ) {
          varyingBandCount++;
          scalarBand = band;
        }
      }
    }
    if (slices.length > 0 && varyingBandCount <= 1) {
      const sorted = slices
        .map(
          slice =>
            values[(cell * dimensions.sliceCount + slice) * dimensions.bandCount + scalarBand]
        )
        .sort((first, second) => first - second);
      const lower = sorted[Math.floor((sorted.length - 1) / 2)];
      const upper = sorted[Math.floor(sorted.length / 2)];
      sliceIndex[cell] = slices.find(slice => {
        const value =
          values[(cell * dimensions.sliceCount + slice) * dimensions.bandCount + scalarBand];
        return value >= lower && value <= upper;
      })!;
    }
    let bestSum = Infinity;
    for (const candidate of varyingBandCount > 1 ? slices : []) {
      let sum = 0;
      for (const other of slices) {
        if (other === candidate) continue;
        let squared = 0;
        for (let band = 0; band < dimensions.bandCount; band++) {
          const difference =
            values[(cell * dimensions.sliceCount + candidate) * dimensions.bandCount + band] -
            values[(cell * dimensions.sliceCount + other) * dimensions.bandCount + band];
          squared += difference * difference;
        }
        sum += Math.sqrt(squared);
      }
      if (sum < bestSum) {
        bestSum = sum;
        sliceIndex[cell] = candidate;
      }
    }
    if (sliceIndex[cell] !== GPU_RASTER_MEDOID_NO_SLICE) {
      status[cell] = 0;
      const start = (cell * dimensions.sliceCount + sliceIndex[cell]) * dimensions.bandCount;
      composite.set(
        values.subarray(start, start + dimensions.bandCount),
        cell * dimensions.bandCount
      );
    }
  }
  return {composite, sliceIndex, status};
}

/** CPU medoid twin, without the GPU's packed-mask and band-loop caps. */
export class CPURasterMedoidCompositeOperation extends GPUCPUOperation {
  constructor(props: RasterMedoidCompositeOperationProps) {
    const id = props.id ?? 'raster-medoid-composite';
    validateOperationGraphNames(
      id,
      {values: props.values, ...(props.mask !== undefined ? {mask: props.mask} : {})},
      {
        composite: props.output.composite,
        sliceIndex: props.output.sliceIndex,
        status: props.output.status
      }
    );
    props = {...props, output: {...props.output}};
    validateCPUCompositeDimensions(id, props);
    super({
      id,
      inputs: {
        values: props.values,
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs: {
        [props.output.composite]: gpuGraph.vector('float32', {
          length: props.cellCount * props.bandCount
        }),
        [props.output.sliceIndex]: gpuGraph.vector('uint32', {length: props.cellCount}),
        [props.output.status]: gpuGraph.vector('uint32', {length: props.cellCount})
      },
      execute({inputs}) {
        const result = computeRasterMedoidCompositeOnCPU(
          getCPUOperationFloat32Rows(id, inputs, 'values'),
          props,
          props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
        );
        return {
          [props.output.composite]: result.composite,
          [props.output.sliceIndex]: result.sliceIndex,
          [props.output.status]: result.status
        };
      }
    });
  }
}
