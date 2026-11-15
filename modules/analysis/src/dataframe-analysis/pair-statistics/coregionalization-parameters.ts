// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {VariogramModelType} from './variogram-model';

/** Fixed basic semivariogram; structurally accepted by local kriging's covariance input. */
export type CoregionalizationStructure = {
  model: VariogramModelType;
  range: number;
  sill: number;
  nugget?: number;
};

/** @internal Model codes shared with the variogram shape and local kriging parameter layout. */
export const COREGIONALIZATION_MODEL_FAMILIES: readonly VariogramModelType[] = [
  'spherical',
  'exponential',
  'gaussian',
  'matern-3/2',
  'matern-5/2'
];

/** Existing local co-kriging layout supports at most four variables. */
export const GPU_COREGIONALIZATION_MAXIMUM_VARIABLE_COUNT = 4;
/** Existing local co-kriging layout supports at most four basic structures. */
export const GPU_COREGIONALIZATION_MAXIMUM_STRUCTURE_COUNT = 4;
/** Reference-tier initialization walks at most this many empirical lag matrices. */
export const GPU_COREGIONALIZATION_MAXIMUM_BIN_COUNT = 256;
/** Bounds graph compilation: two command nodes per sweep. */
export const GPU_COREGIONALIZATION_MAXIMUM_ITERATIONS = 1024;
/** Default relative coefficient-change tolerance, above float32 eigensolver noise. */
export const COREGIONALIZATION_DEFAULT_TOLERANCE = 1e-5;

/** Fixed shape and sweep budget of the LMC fit. */
export type CoregionalizationFitOptions = {
  variableCount: number;
  structureCount: number;
  maximumIterations?: number;
};

/** @internal Shared construction checks; the CPU drops only the GPU bin/iteration limits. */
export function validateCoregionalizationFitOptions(
  id: string,
  options: CoregionalizationFitOptions,
  onGPU: boolean
): void {
  validateCapacity(
    id,
    'variableCount',
    options.variableCount,
    2,
    GPU_COREGIONALIZATION_MAXIMUM_VARIABLE_COUNT
  );
  validateCapacity(
    id,
    'structureCount',
    options.structureCount,
    1,
    GPU_COREGIONALIZATION_MAXIMUM_STRUCTURE_COUNT
  );
  validateCapacity(
    id,
    'maximumIterations',
    options.maximumIterations ?? 128,
    0,
    onGPU ? GPU_COREGIONALIZATION_MAXIMUM_ITERATIONS : Number.MAX_SAFE_INTEGER
  );
}

/**
 * Per-frame parameters: `[relativeTolerance, modelCode, nugget, sill, range, ...]`.
 * Each supplied covariance defines the basic semivariogram `nugget + sill * shape(h/range)`
 * for positive lags. Use sill 1 and nugget 0 for a unit-sill structure, or sill 0 and nugget 1
 * for a pure nugget. Model codes and effective ranges match local kriging.
 */
export function getGPUCoregionalizationFitParameterValues(
  structures: readonly CoregionalizationStructure[],
  tolerance: number = COREGIONALIZATION_DEFAULT_TOLERANCE,
  target: Float32Array = new Float32Array(1 + 4 * structures.length)
): Float32Array {
  const id = 'coregionalization-fit';
  validateCapacity(
    id,
    'structures.length',
    structures.length,
    1,
    GPU_COREGIONALIZATION_MAXIMUM_STRUCTURE_COUNT
  );
  validateFiniteNumber(id, 'tolerance', tolerance, 0, 1);
  validateRequirement(id, Math.fround(tolerance) > 0, 'tolerance must be positive in float32');
  validateRequirement(
    id,
    target.length >= 1 + 4 * structures.length,
    'target must hold 1 + 4 * structures.length elements'
  );
  target[0] = tolerance;
  structures.forEach((structure, index) => {
    validateRequirement(
      id,
      COREGIONALIZATION_MODEL_FAMILIES.includes(structure.model),
      'structures model must name a supported variogram family'
    );
    const values = [structure.nugget ?? 0, structure.sill, structure.range].map(Math.fround);
    validateRequirement(
      id,
      values.every(value => Number.isFinite(Math.fround(value))) &&
        values[0] >= 0 &&
        values[1] >= 0 &&
        values[0] + values[1] > 0 &&
        Number.isFinite(Math.fround(values[0] + values[1])) &&
        Math.fround(values[2]) > 0,
      'structures nugget and sill must be nonnegative with positive total; range must be positive and float32 finite'
    );
    target.set(
      [COREGIONALIZATION_MODEL_FAMILIES.indexOf(structure.model), ...values],
      1 + 4 * index
    );
  });
  return target;
}
