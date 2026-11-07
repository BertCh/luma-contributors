// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/** Ancillary redistribution preserves source support, mask and completion evidence. */
export const COVERAGE_TRANCHE_ONE_OPERATORS: SpatialFamilyOperators = {
  addDasymetricRedistributionRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      targetTotals: 'count',
      targetSupportWeights: 'measure',
      sourceEligibleWeights: 'measure',
      sourceTransferredFractions: 'rate',
      sourceAllocatedTotals: 'count',
      sourceUnallocatedTotals: 'count',
      sourceValidity: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'zone', label: 'source zone', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'operator', name: 'denominator', alternatives: ['zone', 'overlap'], sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Fractional source-cell support, nonnegative ancillary weights and whole-cell target ownership feed explicit areal redistribution. Source totals split into allocated and unallocated mass; missing support and incomplete coverage remain visible instead of manufacturing full allocation.'
    }
  }
};
