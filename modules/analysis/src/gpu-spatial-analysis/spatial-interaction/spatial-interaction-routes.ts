// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';

/** Dense reference kernels scan every pair and retain private origin/destination arrays. */
export const GPU_SPATIAL_INTERACTION_DENSE_MAXIMUM_PAIRS = 4096;

export const GPU_SPATIAL_INTERACTION_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'auto-dense-pair-crossover',
    kind: 'auto-crossover',
    quantity: 'pairCount',
    unit: 'rows',
    appliesWhen: 'method auto, constraint doubly, calibration grid',
    maximum: GPU_SPATIAL_INTERACTION_DENSE_MAXIMUM_PAIRS,
    scope: 'domain',
    alternative: "method: 'sparse' for doubly constrained grid fits",
    accuracyImplications:
      'Both routes fit the same Poisson objective; sorted compensated reductions may differ at float32 tolerance boundaries (compare predictions with 5e-4 relative tolerance).',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction.ts',
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialInteraction'
    }
  }
];
