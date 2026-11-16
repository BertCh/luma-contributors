// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisParameterComparison,
  GPUAnalysisParameterCondition
} from './analysis-experience-types';

/**
 * Evaluates descriptor conditions everywhere they are consumed. An absent condition is active;
 * an empty anyOf is false. Missing values and lists never equal scalars. Comparisons are strict:
 * false, zero and the empty string are values, and numeric strings are not numbers.
 */
export function evaluateGPUAnalysisParameterCondition(
  condition: GPUAnalysisParameterCondition | undefined,
  parameters: Readonly<Record<string, unknown>> = {}
): boolean {
  if (condition === undefined) return true;
  if ('anyOf' in condition) {
    return condition.anyOf.some(branch =>
      evaluateGPUAnalysisParameterCondition(branch, parameters)
    );
  }
  const parameter = 'parameter' in condition ? condition.parameter : condition.whenParameter;
  const value = parameters[parameter];
  return (
    (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') &&
    condition.equals.includes(value)
  );
}

/** Leaf comparisons, for contract validation, controller dependencies and diagnostic text. */
export function getGPUAnalysisParameterComparisons(
  condition: GPUAnalysisParameterCondition
): readonly GPUAnalysisParameterComparison[] {
  return 'anyOf' in condition
    ? condition.anyOf.flatMap(getGPUAnalysisParameterComparisons)
    : [condition];
}
