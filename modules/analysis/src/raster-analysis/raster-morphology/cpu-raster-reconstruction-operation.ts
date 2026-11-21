// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterOperationArray} from '../raster-grid/raster-operation-band';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  calibrateRasterOperationBand,
  readRasterOperationBand,
  snapshotRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../raster-grid/raster-operation-band';
import type {RasterReconstructionOperationProps} from './gpu-raster-reconstruction-operation';
import {
  validateRasterMorphologyOptions,
  captureRasterMorphologyOperationProps,
  visitRasterMorphologyNeighbors
} from './raster-morphology-operation-utils';

export type CPURasterReconstructionOptions = Pick<
  RasterReconstructionOperationProps,
  'grid' | 'method' | 'connectivity' | 'maximumIterations'
>;
export type CPURasterReconstructionResult = {
  values: Float32Array;
  validity: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
};

/**
 * Computes bounded geodesic grayscale reconstruction. Invalid marker/mask pairs
 * (nonfinite, marker > mask for dilation, marker < mask for erosion) stay NaN and are barriers.
 * Only grid.width * grid.height marker and mask rows are read; surplus rows are ignored.
 *
 * What differs from the GPU contributor:
 * - The GPU command-node budget is not used. maximumIterations (default 64) still bounds
 *   output propagation. iterationCount includes the final verification pass; that pass does
 *   not advance the published values. Finite min/max results are exact float32 input values.
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
  validateRasterOperationArray(id, 'marker', marker, 'Float32Array');
  validateRasterOperationArray(id, 'mask', mask, 'Float32Array');
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
  const maximumIterations = options.maximumIterations ?? 64;
  let converged = false;
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
    if (iterationCount > maximumIterations) {
      converged = !changed;
      break;
    }
    [previous, next] = [next, previous];
    if (!changed) {
      converged = true;
      break;
    }
  }
  return {
    values: Float32Array.from(previous),
    validity: Uint32Array.from(previous, value => (Number.isFinite(value) ? 1 : 0)),
    converged: Uint32Array.of(Number(converged)),
    iterationLimitReached: Uint32Array.of(Number(!converged)),
    iterationCount: Uint32Array.of(iterationCount)
  };
}

/**
 * CPU reconstruction with the GPU propagation budget and verification count. Band values and
 * validity may have spare capacity; calibration reads only the grid prefix at their original offsets.
 * Band samples, scale and offset use float32 conversion before multiplication/addition, as on the GPU.
 */
export class CPURasterReconstructionOperation extends GPUCPUOperation {
  constructor(props: RasterReconstructionOperationProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'marker',
      'mask',
      'method',
      'connectivity',
      'maximumIterations',
      'output'
    ]);
    const output = snapshotConstructionFields(props.output, ['values', 'validity', 'status']);
    props = {
      ...props,
      output: Object.freeze({
        ...output,
        status: snapshotConstructionFields(output.status, [
          'converged',
          'iterationCount',
          'iterationLimitReached'
        ])
      })
    };
    props = captureRasterMorphologyOperationProps(
      {
        ...props,
        marker:
          typeof props.marker === 'string'
            ? props.marker
            : snapshotRasterOperationBand(props.marker),
        mask: typeof props.mask === 'string' ? props.mask : snapshotRasterOperationBand(props.mask)
      },
      'reconstruction'
    );
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
    const inputs = {
      marker: typeof props.marker === 'string' ? props.marker : props.marker.storage.values,
      mask: typeof props.mask === 'string' ? props.mask : props.mask.storage.values,
      ...(typeof props.marker !== 'string' && props.marker.validity !== undefined
        ? {markerValidity: props.marker.validity}
        : {}),
      ...(typeof props.mask !== 'string' && props.mask.validity !== undefined
        ? {maskValidity: props.mask.validity}
        : {})
    };
    validateOperationGraphNames(id, inputs, {
      values: props.output.values,
      validity: props.output.validity,
      'status.converged': props.output.status.converged,
      ...Object.fromEntries(
        Object.entries(props.output.status)
          .filter(([, value]) => value !== undefined)
          .map(([port, value]) => [`status.${port}`, value])
      )
    });
    super({
      id,
      inputs,
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
            : calibrateReconstructionBand(
                id,
                'marker',
                readRasterOperationBand(
                  id,
                  {
                    values: inputs['marker'],
                    ...(props.marker.validity !== undefined
                      ? {validity: inputs['markerValidity']}
                      : {})
                  },
                  props.marker
                ),
                length
              ),
          typeof props.mask === 'string'
            ? getCPUOperationFloat32Rows(id, inputs, 'mask')
            : calibrateReconstructionBand(
                id,
                'mask',
                readRasterOperationBand(
                  id,
                  {
                    values: inputs['mask'],
                    ...(props.mask.validity !== undefined ? {validity: inputs['maskValidity']} : {})
                  },
                  props.mask
                ),
                length
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

/** Calibrates borrowed grid-prefix views without rebasing their values or validity offsets. */
function calibrateReconstructionBand(
  id: string,
  port: 'marker' | 'mask',
  band: CPURasterOperationBand,
  count: number
): Float32Array {
  validateRequirement(
    id,
    band.values.length >= count,
    `${port}.values must hold at least ${count} samples for grid.width * grid.height`
  );
  validateRequirement(
    id,
    !band.validity || band.validity.length >= count,
    `${port}.validity must hold at least ${count} flags for grid.width * grid.height`
  );
  const metadata = {
    id: band.id,
    scale: band.scale,
    offset: band.offset,
    noDataValue: band.noDataValue,
    validity: band.validity?.subarray(0, count)
  };
  // Narrow before slicing so each format remains correlated with its scalar array type.
  const prefix: CPURasterOperationBand =
    band.format === 'float32'
      ? {...metadata, format: band.format, values: band.values.subarray(0, count)}
      : band.format === 'sint32'
        ? {...metadata, format: band.format, values: band.values.subarray(0, count)}
        : {...metadata, format: band.format, values: band.values.subarray(0, count)};
  return Float32Array.from(calibrateRasterOperationBand(prefix, count, true));
}
