// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AnalysisValidationIdentity} from '../../analysis-core/kernels/analysis-validation-error';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  MAXIMUM_REFINEMENT_SHADER_MAGNITUDE,
  MINIMUM_REFINEMENT_SHADER_POSITIVE_VALUE,
  validateRefinementShaderNumber
} from './refinement-storage-validation';

/** Adds the received scalar to the shared named-limit diagnostic, including float32 conversion. */
export function validateRefinementNumber(
  id: AnalysisValidationIdentity,
  path: string,
  value: number,
  positive = false
): void {
  const minimum = positive ? MINIMUM_REFINEMENT_SHADER_POSITIVE_VALUE : 0;
  validateRequirement(
    id,
    typeof value === 'number' &&
      Number.isFinite(Math.fround(value)) &&
      Math.fround(value) >= minimum &&
      Math.fround(value) <= MAXIMUM_REFINEMENT_SHADER_MAGNITUDE,
    `${path} must be finite, ${positive ? 'positive' : 'nonnegative'} and representable in float32 in [${minimum}, ${MAXIMUM_REFINEMENT_SHADER_MAGNITUDE}]; received ${String(value)}`,
    {
      code: 'NUMBER_RANGE',
      path,
      received: value,
      expected: {minimum, maximum: MAXIMUM_REFINEMENT_SHADER_MAGNITUDE, precision: 'float32'}
    }
  );
  validateRefinementShaderNumber(typeof id === 'string' ? id : id.id, path, value, positive);
}

/** Keeps the literal port and the row relationship visible in multi-layer graph diagnostics. */
export function validateRefinementOutputLength(
  id: AnalysisValidationIdentity,
  path: string,
  view: {length: number},
  expected: number,
  source: string,
  minimum = false
): void {
  validateRequirement(
    id,
    minimum ? view.length >= expected : view.length === expected,
    `${path} must have ${minimum ? 'at least ' : ''}${expected} rows to match ${source}; received ${view.length}`,
    {
      code: 'VIEW_LENGTH',
      path,
      received: view.length,
      expected: {[minimum ? 'minimumRows' : 'rows']: expected, source}
    }
  );
}
