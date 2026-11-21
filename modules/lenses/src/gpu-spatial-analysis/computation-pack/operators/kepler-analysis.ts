// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/** Explicit bounded contracts for the Kepler/GeoDa analysis compositions. */
export const KEPLER_ANALYSIS_OPERATORS: SpatialFamilyOperators = {
  GPUQuantileLISA: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'count'},
    variables: {
      membership: 'category',
      boundaries: 'measure',
      pseudoPValues: 'probability',
      validCount: 'count',
      invalidQuantile: 'category',
      classes: 'category'
    },
    stages: ['operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'region', indexSpace: 'csr'}],
    guards: ['uncorrected-p', 'count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'quantileParameters', sweepable: true},
      {stage: 'inference', name: 'permutationCount', sweepable: true},
      {
        stage: 'inference',
        name: 'alternative',
        sweepable: true,
        alternatives: ['greater', 'two-sided', 'geoda-legacy']
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'The spatial-autocorrelation contributor combines type-5 GPUColumnQuantiles, lower-inclusive selected-class membership and GPULocalPermutationTest join counts.'
    },
    metadata: {
      quantileConvention: 'type-5',
      boundaryConvention: 'lower-inclusive',
      pCorrection: 'none',
      permutationTail: 'caller-selected',
      legacyPermutationTail: 'strict-complement',
      legacyPermutationFormula: '(min(upper, P-upper)+1)/(P+1)',
      inclusiveTieRule: 'greater and two-sided return 1 when all draws tie'
    }
  },
  GPUDorlingCartogram: {
    family: 'geometry',
    descriptor: {scope: 'pairwise', scopeSecondary: 'solver', reduction: 'solve'},
    variables: {
      positions: 'coordinate',
      radii: 'measure',
      sourceIds: 'category',
      metrics: 'measure',
      status: 'category'
    },
    stages: ['operator', 'presentation'],
    datumKinds: [{kind: 'circle', label: 'circle', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'maximumIterations', sweepable: false},
      {stage: 'operator', name: 'overlapTolerance', sweepable: true},
      {stage: 'operator', name: 'areaScale', sweepable: true},
      {stage: 'operator', name: 'attraction', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Bounded iterative circle displacement heuristic with penetration and displacement diagnostics; see the command graph.'
    },
    metadata: {
      coordinateModel: 'projected',
      maximumRowCount: 1024,
      maximumIterations: 256,
      semantics: 'heuristic'
    }
  },
  addExcessRiskRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', scopeSecondary: 'local', reduction: 'weighted-mean'},
    variables: {events: 'count', populations: 'count', excessRisk: 'rate'},
    stages: ['operator'],
    datumKinds: [{kind: 'region', label: 'region', indexSpace: '1d'}],
    guards: ['count-vs-rate'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Composes GPUEmpiricalBayesRates to expose observed events divided by expected events at the pooled rate.'
    },
    metadata: {mappedVariableExtent: 'intensive'}
  },
  addSpatialRateRecipe: {
    family: 'recipes',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {events: 'count', populations: 'count', spatialRates: 'rate'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'region', indexSpace: 'csr'}],
    guards: ['count-vs-rate'],
    choices: [
      {
        stage: 'operator',
        name: 'islandPolicy',
        alternatives: ['undefined', 'own-rate'],
        sweepable: false
      },
      {stage: 'operator', name: 'scale', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composes validated binary neighborhood event and population pooling with an explicit island policy.'
    },
    metadata: {weightsStandardisation: 'binary', mappedVariableExtent: 'intensive'}
  }
};
