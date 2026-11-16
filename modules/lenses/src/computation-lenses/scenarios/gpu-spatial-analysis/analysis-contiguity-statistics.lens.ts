// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-spatial-analysis.analysis-contiguity-statistics',
  element: 'GPUAnalysisContiguityStatistics',
  title: 'Queen cardinalities of two adjacent squares and one island',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 'synthetic-analysis-profile/1',
    source: 'Deterministic planar points and unit squares authored for these examples',
    licence: 'MIT (repository-authored synthetic fixture; no third-party data)'
  },
  async load() {
    const {runAnalysisProfileExample} = await import('./analysis-profile-run');
    return {run: device => runAnalysisProfileExample(device, 'GPUAnalysisContiguityStatistics')};
  }
});
