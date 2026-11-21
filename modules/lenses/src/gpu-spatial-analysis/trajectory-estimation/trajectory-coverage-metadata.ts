// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../computation-pack/types';

export const TRAJECTORY_COVERAGE_OPERATORS: SpatialFamilyOperators = {
  GPUTrajectoryKalman: {
    family: 'lines',
    descriptor: {scope: 'solver', scopeSecondary: 'zonal', reduction: 'none'},
    variables: {
      filteredState: 'coordinate',
      filteredCovariance: 'measure',
      smoothedState: 'coordinate',
      smoothedCovariance: 'measure',
      status: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'observation', label: 'Track observation', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'operator', name: 'parameters', sweepable: true}],
    metadata: {
      coordinateModel: 'projected planar',
      temporalModel: 'relative float32 seconds',
      uncertainty: 'isotropic or full correlated 4x4 covariance',
      solver: 'Kalman/Joseph and RTS'
    },
    formula: {
      formula: 'none',
      reason:
        'Forward state/covariance prediction and Joseph measurement updates precede a backward RTS solve; missing observations, gap resets and RTS fallback (bit 8) after a singular prediction or nonfinite backward correction is explicit.'
    }
  },
  GPUTrajectoryRestriction: {
    family: 'lines',
    descriptor: {scope: 'local', scopeSecondary: 'zonal', reduction: 'select'},
    variables: {
      intervals: 'measure',
      positions: 'coordinate',
      status: 'category',
      relationships: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'segment', label: 'Source trajectory segment', indexSpace: 'csr'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'operator', name: 'parameters', sweepable: true}],
    metadata: {
      coordinateModel: 'planar',
      interpolation: 'linear',
      boundaries: 'closed/open/half-open',
      geometry: 'rectangle, polygon rings or pairwise within-distance'
    },
    formula: {
      formula: 'none',
      reason:
        'Time clipping and slab intersection preserve source-segment intervals; three-valued ever/always reductions preserve gap and empty-support uncertainty.'
    }
  },
  GPUTrackSimilaritySearch: {
    family: 'lines',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {
      distance: 'measure',
      lowerBound: 'measure',
      similarity: 'measure',
      upperBound: 'measure',
      matched: 'category',
      status: 'category'
    },
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'pair', label: 'Query/candidate track pair', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'metric',
        alternatives: ['dynamic-time-warping', 'frechet', 'longest-common-subsequence'],
        sweepable: false
      },
      {stage: 'operator', name: 'threshold', sweepable: true}
    ],
    metadata: {
      coordinateModel: 'planar',
      bound: 'endpoint distance or possible-match cardinality',
      search: 'exact threshold and stable per-query top-k',
      capacity: 'explicit pair batch and compacted survivor budget'
    },
    formula: {
      formula: 'none',
      reason:
        'An admissible endpoint bound excludes impossible pairs before invoking existing tiled DTW/Frechet programs; an admissible cardinality upper bound precedes LCSS; exact evaluated and bound-pruned results have distinct status.'
    }
  }
};
