// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';

/**
 * Largest row count for which an omitted `mode` selects `'exact'`.
 *
 * Exact mode runs one unit-vector probe per row. Its work is `rows * maximumPower * nnz` and
 * its probe batches (scratch capped at 64 MiB) number `ceil(rows * rows / 8M)`. For an 8-neighbor
 * `W` at `maximumPower = 256`, 4096 rows take 2 batches, about 520 dispatches and `3e10`
 * multiply-adds. At the 16384-row forced exact limit this grows to 32 batches, about 8200 dispatches
 * and `5e11` multiply-adds.
 */
export const GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS = 4096;

export const GPU_SPATIAL_WEIGHTS_TRACES_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'auto-exact-crossover',
    kind: 'auto-crossover',
    quantity: 'rows',
    unit: 'rows',
    appliesWhen: 'mode omitted',
    maximum: GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS,
    scope: 'domain',
    alternative: "mode: 'stochastic'",
    accuracyImplications:
      'Exact uses unit probes and float32 rounding only; stochastic uses seeded Hutchinson estimates and publishes approximation=1.',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces.ts',
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialWeightsTraces'
    }
  }
];
