// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';

/** Largest group count `reduction: 'auto'` reduces without sorting. */
export const GPU_GEOGRAPHIC_DISTRIBUTION_AUTO_CHUNKED_MAXIMUM_GROUPS = 64;

export const GPU_GEOGRAPHIC_DISTRIBUTION_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'auto-chunked-crossover',
    kind: 'auto-crossover',
    quantity: 'groupCount',
    unit: 'elements',
    appliesWhen: 'reduction auto',
    maximum: GPU_GEOGRAPHIC_DISTRIBUTION_AUTO_CHUNKED_MAXIMUM_GROUPS,
    scope: 'domain',
    alternative: "reduction: 'sorted'",
    accuracyImplications:
      'Both routes compute the same moments and Vardi-Zhang median iteration; float32 reduction order may change roundoff and convergence at tolerance boundaries.',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution.ts',
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUGeographicDistribution'
    }
  }
];
