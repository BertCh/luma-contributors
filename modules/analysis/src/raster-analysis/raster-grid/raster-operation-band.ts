// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPURasterBufferBand,
  GPURasterMetadata,
  GPURasterScalarFormat
} from '@luma.gl/graph/gpu-raster';
import type {GPUContributorOperationViewGetter} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationInt32Rows,
  getCPUOperationUint32Rows,
  type CPUOperationInputs
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Buffer band whose storage and validity are logical graph value names. @internal */
export type RasterOperationBand = {
  id: string;
  format: GPURasterScalarFormat;
  storage: {kind: 'buffer'; values: string};
  validity?: string;
  scale?: number;
  offset?: number;
  noDataValue?: number;
};

/** Pure-function band metadata and borrowed input rows. @internal */
export type CPURasterOperationBand = Omit<RasterOperationBand, 'storage' | 'validity'> & {
  values: Float32Array | Int32Array | Uint32Array;
  validity?: Uint32Array;
};

/** Resolves the input band's ports without changing calibration or nodata metadata. @internal */
export function resolveRasterOperationBand(
  input: RasterOperationBand,
  getView: GPUContributorOperationViewGetter
): GPURasterBufferBand {
  return {
    ...input,
    storage: {kind: 'buffer', values: getView('values')},
    validity: input.validity !== undefined ? getView<'uint32'>('validity') : undefined
  } as GPURasterBufferBand;
}

/** Reads packed rows in the band's exact scalar format. @internal */
export function readRasterOperationBand(
  id: string,
  inputs: CPUOperationInputs,
  input: RasterOperationBand
): CPURasterOperationBand {
  const readRows =
    input.format === 'float32'
      ? getCPUOperationFloat32Rows
      : input.format === 'uint32'
        ? getCPUOperationUint32Rows
        : getCPUOperationInt32Rows;
  return {
    ...input,
    values: readRows(id, inputs, 'values'),
    validity:
      input.validity !== undefined ? getCPUOperationUint32Rows(id, inputs, 'validity') : undefined
  };
}

/** Construction-time scalar metadata checks, shared by the raster CPU twins. @internal */
export function validateRasterOperationBand(
  id: string,
  input: RasterOperationBand | CPURasterOperationBand,
  grid: GPURasterMetadata
): void {
  validateCapacity(id, 'grid.width', grid.width);
  validateCapacity(id, 'grid.height', grid.height);
  validateCapacity(id, 'grid cells', grid.width * grid.height);
  validateRequirement(
    id,
    grid.affine.length === 6 && grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite values'
  );
  validateRequirement(id, Boolean(input.id), 'input.id must be nonempty');
  validateRequirement(
    id,
    ['float32', 'uint32', 'sint32'].includes(input.format),
    'input.format must be float32, uint32 or sint32'
  );
  validateRequirement(
    id,
    Number.isFinite(input.scale ?? 1) && Number.isFinite(input.offset ?? 0),
    'input.scale and input.offset must be finite'
  );
}

/** Calibrates in float64, comparing nodata in the band's scalar representation. @internal */
export function calibrateRasterOperationBand(
  input: CPURasterOperationBand,
  count: number
): Float64Array {
  if (input.values.length !== count)
    throw new Error('input.values must contain exactly one sample per pixel');
  if (input.validity && input.validity.length !== count)
    throw new Error('input.validity must contain exactly one flag per pixel');
  // WGSL float32 scalar literals round the sentinel; integer sentinels remain exact.
  const noDataValue =
    input.format === 'float32' && input.noDataValue !== undefined
      ? Math.fround(input.noDataValue)
      : input.noDataValue;
  return Float64Array.from({length: count}, (_, cell) => {
    const raw = input.values[cell];
    const value = raw * (input.scale ?? 1) + (input.offset ?? 0);
    return (input.validity?.[cell] ?? 1) !== 0 &&
      raw !== noDataValue &&
      Number.isFinite(raw) &&
      Number.isFinite(value)
      ? value
      : NaN;
  });
}
