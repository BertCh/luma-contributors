// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';
defineLensScenario({
  id: 'gpu-spatial-analysis.nested-interpolation-validation',
  element: 'GPUNestedInterpolationValidation',
  title: 'Nested interpolation: training-only parameter selection',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'synthetic-nested-validation-twelve-sites',
    source: 'Repository-authored twelve planar sites; two outer folds and inner block seed2.',
    licence: 'MIT'
  },
  async load() {
    const {runNestedInterpolationValidationLens} = await import(
      './nested-interpolation-validation.run'
    );
    return {run: runNestedInterpolationValidationLens};
  }
});
