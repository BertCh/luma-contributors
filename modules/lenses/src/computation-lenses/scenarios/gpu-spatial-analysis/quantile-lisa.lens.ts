// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-spatial-analysis.quantile-lisa',
  element: 'GPUQuantileLISA',
  title: 'Selected upper-quartile local join counts',
  audience: 'engineer',
  rungs: DEFAULT_LENS_RUNGS,
  defaultRung: 'graph',
  dataset: {
    id: 'quantile-lisa-cycle/1',
    source:
      'Repository-authored eight-row cycle with neighboring upper-quartile values; illustrative, not observations.',
    licence: 'MIT (repository-authored illustrative inputs)'
  },
  async load() {
    const [{runS2Lens}, {buildQuantileLISAScenario}] = await Promise.all([
      import('./s2-lens-run'),
      import('./quantile-lisa-examples')
    ]);
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.quantile-lisa',
          element: 'GPUQuantileLISA',
          datasets: ['quantile-lisa-cycle/1'],
          seed: 314159,
          choices: {quantileParameters: 'quantile 4 of 4', permutationCount: 199},
          build: buildQuantileLISAScenario
        })
    };
  }
});
