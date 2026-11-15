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
  },
  GPUCoregionalizationFit: {
    family: 'pair-statistics',
    descriptor: {scope: 'solver', reduction: 'solve'},
    stages: ['filter', 'aggregate', 'summarise'],
    datumKinds: [
      {kind: 'coefficient', label: 'coregionalization matrix coefficient', indexSpace: '1d'}
    ],
    guards: [],
    choices: ['variableCount', 'structureCount', 'maximumIterations', 'tolerance'],
    formula: {
      id: 'gpu-dataframe:GPUCoregionalizationFit',
      derivation: 'curated',
      tex: String.raw`\term{matrices}{(B_s)_s}=\operatorname*{argmin}_{B_s\succeq0}\frac{\sum_h N_h\lVert\Gamma_h-\sum_s B_s g_s(h)\rVert_F^2}{\sum_h N_h},\quad B_s=L_sL_s^T`,
      ascii:
        'Pair-weighted Frobenius least squares of complete direct/cross variogram matrices with fixed basic semivariograms g_s(h) = nugget_s + sill_s * shape_s(h/range_s). Off-diagonal errors count twice. Sequential residual fits project each B_s onto the PSD cone from a zero start; a budget exit publishes the current PSD fit. Spherical, exponential, Gaussian and Matern 3/2 and 5/2 use effective ranges; small-argument series preserve Matern shapes.',
      symbols: [{id: 'matrices', role: 'output', binding: 'wgsl:matrices'}]
    },
    variables: {
      distances: {level: 'ratio', extent: 'intensive'},
      semivariances: {level: 'interval', extent: 'intensive'},
      pairCounts: {level: 'ratio', extent: 'extensive'},
      parameters: {level: 'interval', extent: 'intensive'},
      matrices: {level: 'interval', extent: 'intensive'},
      loadings: {level: 'interval', extent: 'intensive'},
      diagnostics: {level: 'interval', extent: 'intensive'},
      validity: {level: 'nominal'},
      iterationCount: {level: 'ratio', extent: 'intensive'},
      converged: {level: 'nominal'},
      iterationLimitReached: {level: 'nominal'},
      invalidCount: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: [
      'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts'
    ],
    metadata: {
      gpuResident: true,
      tier: 'reference',
      maximumVariableCount: 4,
      maximumStructureCount: 4,
      maximumBinCount: 256,
      defaultMaximumIterations: 128
    }
  }
};
