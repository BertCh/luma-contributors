// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';

/** Pairwise ranking compares every pair and is retained as the small reference route. */
export const GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS = 64;

export const GPU_RANK_MARKOV_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'auto-pairwise-crossover',
    kind: 'auto-crossover',
    quantity: 'rowCount',
    unit: 'rows',
    appliesWhen: 'rankingMethod auto',
    maximum: GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS,
    scope: 'domain',
    alternative: "rankingMethod: 'sorted' without bounded diagnostics",
    accuracyImplications:
      'Both routes publish the same ordinal ranks and exact integer transition counts; mobility differs only by float32 reduction rounding.',
    source: 'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts',
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPURankMarkov'
    }
  }
];
