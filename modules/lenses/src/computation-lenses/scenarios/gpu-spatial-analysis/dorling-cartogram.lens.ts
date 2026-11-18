// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, DEFAULT_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-spatial-analysis.dorling-cartogram',
  element: 'GPUDorlingCartogram',
  title: 'Coincident planar circles with area-derived radii',
  audience: 'engineer',
  rungs: DEFAULT_LENS_RUNGS,
  defaultRung: 'graph',
  dataset: {
    id: 'dorling-illustrative/1',
    source:
      'Repository-authored coincident pair in arbitrary planar units; values pi give unit radii.',
    licence: 'MIT (repository-authored illustrative inputs)'
  },
  async load() {
    const [{runS2Lens}, {buildDorlingCartogramScenario}] = await Promise.all([
      import('./s2-lens-run'),
      import('./dorling-cartogram-examples')
    ]);
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.dorling-cartogram',
          element: 'GPUDorlingCartogram',
          datasets: ['dorling-illustrative/1'],
          choices: {maximumIterations: 8, overlapTolerance: 0.001, areaScale: 1, attraction: 0.01},
          build: buildDorlingCartogramScenario
        })
    };
  }
});
