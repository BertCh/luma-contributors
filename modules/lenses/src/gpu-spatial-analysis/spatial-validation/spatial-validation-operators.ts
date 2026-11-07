// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../computation-pack/types';

export const NESTED_VALIDATION_OPERATORS: SpatialFamilyOperators = {
  GPUNestedInterpolationValidation: {
    family: 'recipes',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {predictions: 'score', scores: 'score', selected: 'category', status: 'category'},
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'fold', label: 'Outer heldout fold', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'innerFoldCount', stage: 'operator', sweepable: false},
      {name: 'neighborCount', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      id: 'gpu-spatial-analysis.nested-interpolation-validation',
      derivation: 'curated',
      tex: String.raw`\hat c_f=\arg\min_c\sum_{g}\sum_{i\in V_{fg}}(\hat z_{c,T_{fg}}(x_i)-z_i)^2,\quad T_{fg}\subset T_f`,
      ascii:
        'For each outer fold, fit each supplied covariance or RBF bandwidth candidate only on inner training rows; select complete minimum inner heldout MSE; refit on outer training and score outer testing.',
      symbols: [{id: 'scores', role: 'output', binding: 'wgsl:scores'}]
    },
    metadata: {
      backlog: 'ST10',
      maximumRows: 256,
      maximumOuterFolds: 16,
      maximumInnerFolds: 8,
      maximumCandidates: 16,
      maximumNeighborCount: 32,
      preprocessing: 'training-only',
      selection: 'nested complete inner heldout MSE',
      ties: 'lowest candidate index'
    }
  }
};
