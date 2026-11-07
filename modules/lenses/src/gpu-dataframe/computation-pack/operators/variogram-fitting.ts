// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Resident constrained variogram fitting and model-selection evidence. */
export const VARIOGRAM_FITTING_OPERATORS: DataFrameFamilyOperators = {
  GPUVariogramModelFit: {
    family: 'pair-statistics',
    descriptor: {scope: 'solver', reduction: 'solve'},
    stages: ['filter', 'aggregate', 'summarise'],
    datumKinds: [{kind: 'model', label: 'variogram model', indexSpace: '1d'}],
    guards: [],
    choices: ['models', 'weighting', 'rangeSteps', 'maximumIterations'],
    formula: {
      id: 'gpu-dataframe:GPUVariogramModelFit',
      derivation: 'curated',
      tex: String.raw`\term{parameters}{(c_0,c,a)}=\operatorname*{argmin}_{c_0,c\ge0,a>0}\sum_h w_h(\gamma_h-c_0-c\,g(h/a))^2`,
      ascii:
        'Constrained nugget/sill at each log-range candidate; golden-section refinement; pair/unweighted/Cressie weights. Boundary and exhausted refinement have status 3.',
      symbols: [{id: 'parameters', role: 'output', binding: 'wgsl:parameters'}]
    },
    variables: {
      parameters: {level: 'ratio', extent: 'intensive'},
      diagnostics: {level: 'interval', extent: 'intensive'},
      statuses: {level: 'nominal'}
    },
    sourceFiles: [
      'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit.ts'
    ],
    metadata: {gpuResident: true, defaultRangeSteps: 200, defaultMaximumIterations: 32}
  }
};
