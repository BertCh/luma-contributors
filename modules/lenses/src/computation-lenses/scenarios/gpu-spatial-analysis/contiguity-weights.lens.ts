// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario, FORMULA_LENS_RUNGS} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-spatial-analysis.contiguity-weights',
  element: 'GPUContiguityWeights',
  title: 'Queen contiguity between Chicago tracts',
  rungs: FORMULA_LENS_RUNGS,
  dataset: {
    id: 's1-chicago-tracts/1',
    source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022)',
    licence:
      'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain; CDC/ATSDR SVI 2022, public domain'
  },
  async load() {
    const [{runExample}, {loadChicagoTracts}, {runTractWeightsScenario}, {WEIGHTS_OPERATORS}] =
      await Promise.all([
        import('./contiguity-weights.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/weights')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        runTractWeightsScenario({
          device,
          id: 'gpu-spatial-analysis.contiguity-weights',
          tracts,
          entry: WEIGHTS_OPERATORS['GPUContiguityWeights'],
          presentation: 'graph',
          choices: {
            criterion: 'queen',
            snapTolerance: 0
          },
          metadata: {weightsStandardisation: 'binary'},
          runExample
        })
    };
  }
});
