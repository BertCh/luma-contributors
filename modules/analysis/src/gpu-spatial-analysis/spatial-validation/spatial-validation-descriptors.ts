// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUSpatialAnalysisCapability} from '../ontology/spatial-analysis-ontology';
export const NESTED_VALIDATION_CAPABILITY: GPUSpatialAnalysisCapability = {
  id: 'nested-interpolation-validation',
  title: 'Nested interpolation model selection',
  status: 'available',
  domain: 'spatial-models',
  stage: 'infer',
  summary:
    'Leakage-safe resident variogram/covariance and RBF bandwidth selection, refitting and heldout metrics over explicit spatial folds.',
  inputs: ['points', 'attributes', 'parameters'],
  outputs: ['attributes', 'summary'],
  coordinateModels: ['planar'],
  outputBound: 'source-sized',
  operators: ['GPUNestedInterpolationValidation'],
  limitation:
    'Explicit finite candidate ladders and bounded local systems; outer masks must be disjoint. Missing inner predictions invalidate candidate scores.',
  evidence: {
    representativeSourceRows: 12,
    transientMemory: 'solver-matrix',
    dispatch: 'multi-pass',
    correctnessOracle:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-validation-advanced.spec.ts'
  }
};
export const NESTED_VALIDATION_OPERATOR_OVERRIDE = {
  GPUNestedInterpolationValidation: {
    contract: 'GPUNestedInterpolationValidationProps',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/spatial-validation-advanced.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/spatial-validation-advanced.md',
    evidence: 'partial' as const
  }
};
