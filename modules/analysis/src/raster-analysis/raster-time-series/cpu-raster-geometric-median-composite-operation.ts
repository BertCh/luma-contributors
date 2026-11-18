// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH} from './gpu-raster-geometric-median-composite';
import type {RasterGeometricMedianCompositeOperationProps} from './gpu-raster-geometric-median-composite-operation';
import {getGPURasterGeometricMedianCompositeOutputLayout} from './raster-geometric-median-output-layout';
import {
  getCPUCompositeSlices,
  validateCPUCompositeArrays,
  validateCPUCompositeDimensions,
  type CPUCompositeDimensions
} from './cpu-composite-observations';

export type CPURasterGeometricMedianCompositeOptions = CPUCompositeDimensions & {
  iterationCount: number;
};
export type CPURasterGeometricMedianCompositeResult = {
  composite: Float32Array;
  status: Uint32Array;
  lastStep: Float32Array;
};

/**
 * Runs exactly `iterationCount` epsilon-clamped Weiszfeld passes from the valid-observation mean,
 * with the contributor's packed `[distanceEpsilon]` per-frame parameters and dense output layout.
 * A zero observation mask or any nonfinite band rejects the whole slice.
 *
 * What differs from the GPU contributor:
 * - Means, weights and updates use float64, rounded once in composite and lastStep outputs.
 *   The requested iteration count and epsilon clamp are retained as the documented method.
 * - GPU limits of 256 slices, 16 bands and 64 iterations are dropped; uint32 shape limits remain.
 *
 * @throws If values, mask or parameters are too short, or dimensions/iterationCount invalid.
 */
export function computeRasterGeometricMedianCompositeOnCPU(
  values: Float32Array,
  parameters: Float32Array,
  options: CPURasterGeometricMedianCompositeOptions,
  mask?: Uint32Array
): CPURasterGeometricMedianCompositeResult {
  validateCPUCompositeDimensions('computeRasterGeometricMedianCompositeOnCPU', options);
  validateCapacity(
    'computeRasterGeometricMedianCompositeOnCPU',
    'iterationCount',
    options.iterationCount
  );
  validateCPUCompositeArrays(values, mask, options);
  if (parameters.length < GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH)
    throw new Error('parameters must contain distanceEpsilon');
  // Layout of getGPURasterGeometricMedianCompositeParameterValues; the sole parameter is epsilon.
  const [distanceEpsilon] = parameters;
  const outputLayout = getGPURasterGeometricMedianCompositeOutputLayout({
    ...options,
    id: 'computeRasterGeometricMedianCompositeOnCPU',
    lastStep: true
  });
  const composite = new Float32Array(outputLayout.composite.length).fill(NaN);
  const lastStep = new Float32Array(outputLayout.lastStep!.length).fill(NaN);
  const status = new Uint32Array(outputLayout.status.length).fill(1);
  for (let cell = 0; cell < options.cellCount; cell++) {
    const slices = getCPUCompositeSlices(values, mask, options, cell);
    if (slices.length === 0) continue;
    const estimate = new Float64Array(options.bandCount);
    const numerator = new Float64Array(options.bandCount);
    for (const slice of slices)
      for (let band = 0; band < options.bandCount; band++)
        estimate[band] += values[(cell * options.sliceCount + slice) * options.bandCount + band];
    for (let band = 0; band < options.bandCount; band++) estimate[band] /= slices.length;
    let step = 0;
    for (let iteration = 0; iteration < options.iterationCount; iteration++) {
      numerator.fill(0);
      let weightSum = 0;
      for (const slice of slices) {
        const base = (cell * options.sliceCount + slice) * options.bandCount;
        let squared = 0;
        for (let band = 0; band < options.bandCount; band++)
          squared += (values[base + band] - estimate[band]) ** 2;
        const weight = 1 / Math.max(Math.sqrt(squared), distanceEpsilon);
        weightSum += weight;
        for (let band = 0; band < options.bandCount; band++)
          numerator[band] += weight * values[base + band];
      }
      let squaredStep = 0;
      for (let band = 0; band < options.bandCount; band++) {
        const next = numerator[band] / weightSum;
        squaredStep += (next - estimate[band]) ** 2;
        estimate[band] = next;
      }
      step = Math.sqrt(squaredStep);
    }
    composite.set(estimate, cell * options.bandCount);
    lastStep[cell] = step;
    status[cell] = 0;
  }
  return {composite, status, lastStep};
}

/** CPU twin, preserving fixed-count Weiszfeld semantics without GPU array/iteration caps. */
export class CPURasterGeometricMedianCompositeOperation extends GPUCPUOperation {
  constructor(props: RasterGeometricMedianCompositeOperationProps) {
    const id = props.id ?? 'raster-geometric-median-composite';
    validateCPUCompositeDimensions(id, props);
    validateCapacity(id, 'iterationCount', props.iterationCount);
    const lastStep = props.output.lastStep;
    const outputLayout = getGPURasterGeometricMedianCompositeOutputLayout({
      ...props,
      id,
      lastStep: lastStep !== undefined
    });
    super({
      id,
      inputs: {
        values: props.values,
        parameters: props.parameters,
        ...(props.mask !== undefined ? {mask: props.mask} : {})
      },
      outputs: {
        [props.output.composite]: gpuGraph.vector(outputLayout.composite.format, {
          length: outputLayout.composite.length
        }),
        [props.output.status]: gpuGraph.vector(outputLayout.status.format, {
          length: outputLayout.status.length
        }),
        ...(lastStep !== undefined && outputLayout.lastStep
          ? {
              [lastStep]: gpuGraph.vector(outputLayout.lastStep.format, {
                length: outputLayout.lastStep.length
              })
            }
          : {})
      },
      execute({inputs}) {
        const result = computeRasterGeometricMedianCompositeOnCPU(
          getCPUOperationFloat32Rows(id, inputs, 'values'),
          getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          props,
          props.mask !== undefined ? getCPUOperationUint32Rows(id, inputs, 'mask') : undefined
        );
        return {
          [props.output.composite]: result.composite,
          [props.output.status]: result.status,
          ...(lastStep !== undefined ? {[lastStep]: result.lastStep} : {})
        };
      }
    });
  }
}
