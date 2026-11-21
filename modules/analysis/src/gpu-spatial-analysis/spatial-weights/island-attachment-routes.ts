// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';

export const GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS = 4096;
export const GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS = 65535;

export const GPU_ISLAND_ATTACHMENT_ROUTES: readonly GPUAnalysisRoute[] = [
  {
    name: 'auto-scan-crossover',
    kind: 'auto-crossover',
    quantity: 'rows',
    unit: 'rows',
    appliesWhen: 'nearestSearch auto',
    maximum: GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS,
    scope: 'domain',
    alternative: "nearestSearch: 'grid'",
    accuracyImplications:
      'Both routes use the same nearest comparison; grid falls back when candidates cannot prove the answer.',
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts',
    planner: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUIslandAttachment'
    }
  },
  {
    name: 'forced-scan-maximum',
    kind: 'hard-limit',
    quantity: 'rows for nearestSearch scan',
    unit: 'rows',
    appliesWhen: 'nearestSearch scan',
    maximum: GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS,
    scope: 'domain',
    accuracyImplications: 'Exact scan; grid uses bounded exact fallback.',
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts'
  }
];
