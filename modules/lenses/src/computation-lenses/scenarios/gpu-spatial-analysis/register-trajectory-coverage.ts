// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';
import type {TrajectoryCoverageKind} from './trajectory-coverage.lens';

const scenarios: readonly [TrajectoryCoverageKind, string, string][] = [
  [
    'trajectory-kalman',
    'GPUTrajectoryKalman',
    'Two observations: forward estimates and backward uncertainty'
  ],
  [
    'trajectory-restriction',
    'GPUTrajectoryRestriction',
    'A rectangle crossing and a track without common time support'
  ],
  [
    'track-similarity-search',
    'GPUTrackSimilaritySearch',
    'An exact DTW match beside a safely pruned candidate'
  ]
];
for (const [kind, element, title] of scenarios) {
  defineLensScenario({
    id: `gpu-spatial-analysis.${kind}`,
    entry: 'gpu-spatial-analysis',
    element,
    title,
    audience: 'analyst',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    defaultRung: 'result',
    dataset: {
      id: `synthetic-${kind}-v1`,
      source: 'Deterministic projected trajectory in trajectory-coverage.lens.ts',
      licence: 'MIT'
    },
    load: async () => {
      const module = await import('./trajectory-coverage.lens');
      return {run: device => module.runTrajectoryCoverage(device, kind)};
    }
  });
}
