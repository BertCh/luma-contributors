// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisLimit} from '../spatial-analysis-ontology';

/** Both visibility recipes build the same spatial adjacency before any network metrics. */
export const VISIBILITY_GRAPH_LIMITS: readonly GPUAnalysisLimit[] = [
  {
    name: 'positions.length (csr)',
    maximum: 1000000,
    reason: 'Indexed sparse point queries; device bindings and grid visit budgets also apply.',
    source: 'modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr.ts'
  },
  {
    name: 'edgeCapacity (csr)',
    maximum: 16777216,
    reason: 'Sparse output slots and device scratch bindings.',
    source: 'modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr.ts'
  },
  {
    name: 'positions.length (reference)',
    maximum: 256,
    reason: 'Dense all-pairs reference.',
    source: 'modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph.ts'
  }
];
