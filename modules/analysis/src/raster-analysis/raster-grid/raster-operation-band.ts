// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPURasterBufferBand,
  GPURasterMetadata,
  GPURasterScalarFormat
} from '@luma.gl/graph/gpu-raster';
import {
  validateOperationGraphNames,
  type GPUContributorOperationViewGetter
} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
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

// Read the intrinsic element brand across realms; own Symbol.toStringTag cannot spoof it.
const TYPED_ARRAY_NAME_GETTER = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype),
  Symbol.toStringTag
)?.get;

/** Validate the intrinsic element brand without scanning rows, including across realms. @internal */
export function validateRasterOperationArray(
  id: string,
  name: string,
  values: unknown,
  expectedBrand: 'Float32Array' | 'Int32Array' | 'Uint32Array'
): void {
  const receivedBrand = TYPED_ARRAY_NAME_GETTER?.call(values);
  validateRequirement(
    id,
    receivedBrand === expectedBrand,
    `${name} must be ${expectedBrand}; received ${receivedBrand ?? typeof values}`
  );
}

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

/** Pure-function band metadata and borrowed input rows. */
export type CPURasterOperationBand = Omit<
  RasterOperationBand,
  'storage' | 'validity' | 'format'
> & {
  validity?: Uint32Array;
} & (
    | {format: 'float32'; values: Float32Array}
    | {format: 'sint32'; values: Int32Array}
    | {format: 'uint32'; values: Uint32Array}
  );

/** Capture declared metadata and logical port names, including inherited getters. @internal */
export function snapshotRasterOperationBand(input: RasterOperationBand): RasterOperationBand {
  const captured = snapshotConstructionFields(input, [
    'id',
    'format',
    'storage',
    'validity',
    'scale',
    'offset',
    'noDataValue'
  ]);
  return Object.freeze({
    ...captured,
    storage: snapshotConstructionFields(captured.storage, ['kind', 'values'])
  });
}

/** Validate logical names before keyed declarations can collapse duplicate outputs. @internal */
export function validateRasterOperationOutputNames(
  id: string,
  requiredNames: Readonly<Record<string, string>>,
  optionalNames: Readonly<Record<string, string | undefined>>,
  inputNames: Readonly<Record<string, string>>
): void {
  validateOperationGraphNames(id, inputNames, {
    ...requiredNames,
    ...Object.fromEntries(
      Object.entries(optionalNames)
        .filter(([, name]) => name !== undefined)
        .map(([port, name]) => [port, name!])
    )
  });
}

/** Resolves the input band's ports without changing calibration or nodata metadata. @internal */
export function resolveRasterOperationBand(
  input: RasterOperationBand,
  getView: GPUContributorOperationViewGetter
): GPURasterBufferBand {
  return {
    id: input.id,
    format: input.format,
    scale: input.scale,
    offset: input.offset,
    noDataValue: input.noDataValue,
    storage: {kind: 'buffer', values: getView('values')},
    validity: input.validity !== undefined ? getView<'uint32'>('validity') : undefined
  } as GPURasterBufferBand;
}

/** Reads packed rows in the band's exact scalar format. @internal */
export function readRasterOperationBand(
  id: string,
  inputs: CPUOperationInputs,
  input: RasterOperationBand,
  bandName?: string
): CPURasterOperationBand {
  const valuesPort = bandName === undefined ? 'values' : `${bandName}.values`;
  const validityPort = bandName === undefined ? 'validity' : `${bandName}.validity`;
  const bandInputs = {
    [valuesPort]: inputs['values']!,
    [validityPort]: inputs['validity']!
  };
  const metadata = {
    id: input.id,
    scale: input.scale,
    offset: input.offset,
    noDataValue: input.noDataValue,
    validity:
      input.validity !== undefined
        ? getCPUOperationUint32Rows(id, bandInputs, validityPort)
        : undefined
  };
  switch (input.format) {
    case 'float32':
      return {
        ...metadata,
        format: input.format,
        values: getCPUOperationFloat32Rows(id, bandInputs, valuesPort)
      };
    case 'uint32':
      return {
        ...metadata,
        format: input.format,
        values: getCPUOperationUint32Rows(id, bandInputs, valuesPort)
      };
    case 'sint32':
      return {
        ...metadata,
        format: input.format,
        values: getCPUOperationInt32Rows(id, bandInputs, valuesPort)
      };
    default:
      validateRequirement(
        id,
        false,
        `input.format must be float32, uint32 or sint32 (received ${input.format})`
      );
      throw new Error(`${id} unreachable raster format`);
  }
}

/** Construction-time scalar metadata checks, shared by the raster CPU twins. @internal */
export function validateRasterOperationBand(
  id: string,
  input: RasterOperationBand | CPURasterOperationBand,
  grid: GPURasterMetadata,
  bandName = 'input'
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
  if ('values' in input) {
    const ArrayType =
      input.format === 'float32'
        ? Float32Array
        : input.format === 'uint32'
          ? Uint32Array
          : Int32Array;
    const observedClass = TYPED_ARRAY_NAME_GETTER?.call(input.values);
    validateRequirement(
      id,
      observedClass === ArrayType.name,
      `${bandName}.values must be ${ArrayType.name} for format ${input.format} (received ${observedClass ?? input.values?.constructor?.name ?? typeof input.values})`
    );
    if (input.validity !== undefined) {
      validateRasterOperationArray(id, `${bandName}.validity`, input.validity, 'Uint32Array');
    }
  }
  validateRequirement(
    id,
    Number.isFinite(input.scale ?? 1) && Number.isFinite(input.offset ?? 0),
    'input.scale and input.offset must be finite'
  );
}

/**
 * Calibrates in float64 by default, comparing nodata in the band's scalar representation.
 * float32Calibration converts samples/metadata before rounded multiply/add for GPU canonicalization.
 * Both modes require exact sample and validity lengths. @internal
 */
export function calibrateRasterOperationBand(
  input: CPURasterOperationBand,
  count: number,
  float32Calibration = false
): Float64Array {
  validateRasterOperationArray(
    'calibrateRasterOperationBand',
    'input.values',
    input.values,
    input.format === 'float32'
      ? 'Float32Array'
      : input.format === 'uint32'
        ? 'Uint32Array'
        : 'Int32Array'
  );
  if (input.validity !== undefined) {
    validateRasterOperationArray(
      'calibrateRasterOperationBand',
      'input.validity',
      input.validity,
      'Uint32Array'
    );
  }
  if (input.values.length !== count)
    throw new Error('input.values must contain exactly one sample per pixel');
  if (input.validity && input.validity.length !== count)
    throw new Error('input.validity must contain exactly one flag per pixel');
  // WGSL float32 scalar literals round the sentinel; integer sentinels remain exact.
  const noDataValue =
    input.format === 'float32' && input.noDataValue !== undefined
      ? Math.fround(input.noDataValue)
      : input.noDataValue;
  const scale = float32Calibration ? Math.fround(input.scale ?? 1) : (input.scale ?? 1);
  const offset = float32Calibration ? Math.fround(input.offset ?? 0) : (input.offset ?? 0);
  return Float64Array.from({length: count}, (_, cell) => {
    const raw = input.values[cell];
    const value = float32Calibration
      ? Math.fround(Math.fround(Math.fround(raw) * scale) + offset)
      : raw * scale + offset;
    return (input.validity?.[cell] ?? 1) !== 0 &&
      raw !== noDataValue &&
      Number.isFinite(raw) &&
      Number.isFinite(value)
      ? value
      : NaN;
  });
}
