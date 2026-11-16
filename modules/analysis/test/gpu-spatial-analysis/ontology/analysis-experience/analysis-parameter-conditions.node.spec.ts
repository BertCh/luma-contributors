// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {assignGPUAnalysisFieldRoles} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-data-roles';
import {
  evaluateGPUAnalysisParameterCondition,
  getGPUAnalysisParameterComparisons
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-parameter-conditions';
import type {GPUAnalysisParameterCondition} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';

const ROUND_STYLE: GPUAnalysisParameterCondition = {
  anyOf: [
    {parameter: 'capStyle', equals: ['round']},
    {whenParameter: 'joinStyle', equals: ['round']}
  ]
};

describe('shared parameter conditions', () => {
  it('preserves same-parameter any-value comparisons and legacy field-role comparisons', () => {
    for (const condition of [
      {parameter: 'mode', equals: ['sum', 'mean']},
      {whenParameter: 'mode', equals: ['sum', 'mean']}
    ]) {
      expect(evaluateGPUAnalysisParameterCondition(condition, {mode: 'sum'})).toBe(true);
      expect(evaluateGPUAnalysisParameterCondition(condition, {mode: 'mean'})).toBe(true);
      expect(evaluateGPUAnalysisParameterCondition(condition, {mode: 'count'})).toBe(false);
    }
  });

  it('supports OR across different parameters, nesting and absent controllers', () => {
    expect(evaluateGPUAnalysisParameterCondition(ROUND_STYLE, {capStyle: 'round'})).toBe(true);
    expect(evaluateGPUAnalysisParameterCondition(ROUND_STYLE, {joinStyle: 'round'})).toBe(true);
    expect(
      evaluateGPUAnalysisParameterCondition(ROUND_STYLE, {capStyle: 'flat', joinStyle: 'mitre'})
    ).toBe(false);
    expect(evaluateGPUAnalysisParameterCondition(ROUND_STYLE)).toBe(false);
    expect(
      evaluateGPUAnalysisParameterCondition(
        {anyOf: [{anyOf: []}, ROUND_STYLE]},
        {joinStyle: 'round'}
      )
    ).toBe(true);
    expect(getGPUAnalysisParameterComparisons({anyOf: [ROUND_STYLE]})).toEqual(ROUND_STYLE.anyOf);
  });

  it('treats absent conditions as active and empty alternatives as false', () => {
    expect(evaluateGPUAnalysisParameterCondition(undefined)).toBe(true);
    expect(evaluateGPUAnalysisParameterCondition({anyOf: []})).toBe(false);
    expect(
      evaluateGPUAnalysisParameterCondition({parameter: 'value', equals: []}, {value: 1})
    ).toBe(false);
  });

  it('compares scalars strictly, retaining false, zero and empty strings', () => {
    for (const value of [false, 0, '']) {
      expect(
        evaluateGPUAnalysisParameterCondition({parameter: 'value', equals: [value]}, {value})
      ).toBe(true);
    }
    for (const value of [undefined, null, [1], '1']) {
      expect(
        evaluateGPUAnalysisParameterCondition({parameter: 'value', equals: [1]}, {value})
      ).toBe(false);
    }
  });

  it('uses cross-parameter conditions to require field roles', () => {
    const role = {
      role: 'value',
      label: 'Value',
      accepts: ['number'] as const,
      required: ROUND_STYLE
    };
    expect(assignGPUAnalysisFieldRoles([role], [], {capStyle: 'round'}).unfilledRoles).toEqual([
      'value'
    ]);
    expect(assignGPUAnalysisFieldRoles([role], [], {joinStyle: 'round'}).unfilledRoles).toEqual([
      'value'
    ]);
    expect(
      assignGPUAnalysisFieldRoles([role], [], {capStyle: 'flat', joinStyle: 'mitre'}).unfilledRoles
    ).toEqual([]);
  });
});
