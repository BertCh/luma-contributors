// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  validateRasterOperationBand
} from '../raster-grid/raster-operation-band';
import type {RasterReconstructionOperationProps} from './gpu-raster-reconstruction-operation';
import {
  validateRasterMorphologyOptions,
  visitRasterMorphologyNeighbors
} from './raster-morphology-operation-utils';

export type CPURasterReconstructionOptions = Pick<
  RasterReconstructionOperationProps,
  'grid' | 'method' | 'connectivity'
>;
export type CPURasterReconstructionResult = {
  values: Float32Array;
  validity: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
};

/**
 * Computes geodesic grayscale reconstruction to its fixed point. Invalid marker/mask pairs
 * (nonfinite, marker > mask for dilation, marker < mask for erosion) stay NaN and are barriers.
 * Only grid.width * grid.height marker and mask rows are read; surplus rows are ignored.
 *
 * What differs from the GPU contributor:
 * - maximumIterations and the GPU command-node budget are not used: the exact fixed point is
 *   always returned, converged is 1 and iterationLimitReached is 0.
 * - iterationCount is the number of synchronous rounds needed, including the first unchanged
 *   round; it can exceed the GPU budget. Finite min/max results are exact float32 input values.
 * - Subnormal inputs are retained; GPU implementations may flush them to zero.
 *
 * @throws Error naming marker or mask for an inconsistent length, or an invalid option.
 */
export function computeRasterReconstructionOnCPU(
  marker: Float32Array,
  mask: Float32Array,
  options: CPURasterReconstructionOptions
): CPURasterReconstructionResult {
  const id = 'computeRasterReconstructionOnCPU';
  validateRasterMorphologyOptions(id, options);
  validateRequirement(
    id,
    options.method === undefined || options.method === 'dilation' || options.method === 'erosion',
    'method must be dilation or erosion'
  );
  const {width, height} = options.grid;
  const length = width * height;
  if (marker.length < length) throw new Error(`${id} marker must have at least one row per cell`);
  if (mask.length < length) throw new Error(`${id} mask must have at least one row per cell`);
  const dilation = options.method !== 'erosion';
  let previous = Float64Array.from(marker.subarray(0, length), (value, cell) =>
    Number.isFinite(value) &&
    Number.isFinite(mask[cell]) &&
    (dilation ? value <= mask[cell] : value >= mask[cell])
      ? value
      : NaN
  );
  let next = new Float64Array(length);
  let iterationCount = 0;
  for (;;) {
    let changed = false;
    for (let cell = 0; cell < length; cell++) {
      let value = previous[cell];
      if (Number.isFinite(value)) {
        visitRasterMorphologyNeighbors(cell, width, height, options.connectivity ?? 8, neighbor => {
          const neighborValue = previous[neighbor];
          if (Number.isFinite(neighborValue))
            value = dilation ? Math.max(value, neighborValue) : Math.min(value, neighborValue);
        });
        value = dilation ? Math.min(value, mask[cell]) : Math.max(value, mask[cell]);
        if (value !== previous[cell]) changed = true;
      }
      next[cell] = value;
    }
    iterationCount++;
    [previous, next] = [next, previous];
    if (!changed) break;
  }
  return {
    values: Float32Array.from(previous),
    validity: Uint32Array.from(previous, value => (Number.isFinite(value) ? 1 : 0)),
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    iterationCount: Uint32Array.of(iterationCount)
  };
}

/** CPU reconstruction without a command-node or convergence budget. */
export class CPURasterReconstructionOperation extends GPUCPUOperation {
  constructor(props: RasterReconstructionOperationProps) {
    const id = props.id ?? 'raster-reconstruction';
    validateRasterMorphologyOptions(id, props);
    validateRequirement(
      id,
      props.method === undefined || props.method === 'dilation' || props.method === 'erosion',
      'method must be dilation or erosion'
    );
    const length = props.grid.width * props.grid.height;
    for (const input of [props.marker, props.mask]) {
      if (typeof input !== 'string') validateRasterOperationBand(id, input, props.grid);
    }
    super({
      id,
      inputs: {
        marker: typeof props.marker === 'string' ? props.marker : props.marker.storage.values,
        mask: typeof props.mask === 'string' ? props.mask : props.mask.storage.values,
        ...(typeof props.marker !== 'string' && props.marker.validity
          ? {markerValidity: props.marker.validity}
          : {}),
        ...(typeof props.mask !== 'string' && props.mask.validity
          ? {maskValidity: props.mask.validity}
          : {})
      },
      outputs: {
        [props.output.values]: gpuGraph.vector('float32', {length}),
        [props.output.validity]: gpuGraph.vector('uint32', {length}),
        ...Object.fromEntries(
          Object.values(props.output.status)
            .filter(value => value !== undefined)
            .map(value => [value, gpuGraph.vector('uint32', {length: 1})])
        )
      },
      execute({inputs}) {
        const result = computeRasterReconstructionOnCPU(
          typeof props.marker === 'string'
            ? getCPUOperationFloat32Rows(id, inputs, 'marker')
            : Float32Array.from(
                calibrateRasterOperationBand(
                  readRasterOperationBand(
                    id,
                    {
                      values: inputs['marker'],
                      ...(props.marker.validity ? {validity: inputs['markerValidity']} : {})
                    },
                    props.marker
                  ),
                  length
                )
              ),
          typeof props.mask === 'string'
            ? getCPUOperationFloat32Rows(id, inputs, 'mask')
            : Float32Array.from(
                calibrateRasterOperationBand(
                  readRasterOperationBand(
                    id,
                    {
                      values: inputs['mask'],
                      ...(props.mask.validity ? {validity: inputs['maskValidity']} : {})
                    },
                    props.mask
                  ),
                  length
                )
              ),
          props
        );
        return {
          [props.output.values]: result.values,
          [props.output.validity]: result.validity,
          [props.output.status.converged]: result.converged,
          ...(props.output.status.iterationCount !== undefined
            ? {[props.output.status.iterationCount]: result.iterationCount}
            : {}),
          ...(props.output.status.iterationLimitReached !== undefined
            ? {[props.output.status.iterationLimitReached]: result.iterationLimitReached}
            : {})
        };
      }
    });
  }
}
