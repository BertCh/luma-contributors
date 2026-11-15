// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../computation-pack/types';

/** Scoped contributor discovery metadata for bounded point-process analysis. */
export const POINT_PROCESS_OPERATORS: SpatialFamilyOperators = {
  GPUInhomogeneousPointPattern: {
    family: 'statistics',
    descriptor: {scope: 'pairwise', reduction: 'weighted-mean'},
    variables: {
      positions: 'coordinate',
      intensity: 'rate',
      radii: 'measure',
      k: 'measure',
      pairCorrelation: 'rate',
      status: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {kind: 'point', label: 'event', indexSpace: '1d'},
      {kind: 'radius', label: 'radius', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {name: 'bandwidth', stage: 'neighbourhood', sweepable: true},
      {
        name: 'edgeCorrection',
        stage: 'operator',
        alternatives: ['border', 'translation'],
        sweepable: true
      },
      {name: 'pairCorrelation', stage: 'operator', sweepable: true},
      {
        name: 'selfContribution',
        stage: 'operator',
        alternatives: ['include', 'exclude'],
        sweepable: true
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Polygon kernel-mass estimation followed by border or translation K and annular or Epanechnikov pair correlation; GPU signed polygon overlap and explicit self/zero-correction conventions are documented in point-process.md.'
    },
    metadata: {
      coordinateModel: 'projected',
      edgeCorrection: 'border-or-translation',
      maximumEvents: 4096,
      maximumOverlapEdgePairs: 1048576
    }
  },
  GPUPointPatternEnvelope: {
    family: 'statistics',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {
      observed: 'score',
      simulations: 'score',
      reference: 'score',
      lower: 'score',
      upper: 'score',
      statistics: 'score',
      pValueInterval: 'probability',
      status: 'category'
    },
    stages: ['inference'],
    datumKinds: [{kind: 'radius', label: 'curve ordinate', indexSpace: '1d'}],
    guards: [],
    choices: [
      {
        name: 'statistic',
        stage: 'inference',
        alternatives: ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'],
        sweepable: true
      },
      {name: 'rank', stage: 'inference', sweepable: true},
      {name: 'significanceLevel', stage: 'inference', sweepable: true},
      {name: 'simulationCount', stage: 'inference', sweepable: true},
      {
        name: 'fittedParameters',
        stage: 'inference',
        alternatives: ['none', 'fixed-external', 'refit-each-replicate'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'The selected maximum or integrated-square deviation is compared with supplied exchangeable null replicates using conservative plus-one ranks. Pointwise order statistics are separate outputs.'
    },
    metadata: {invalidReplicatePolicy: 'reject-complete-test', pointwiseEnvelopeIsGlobal: false}
  },
  GPUOPTICS: {
    family: 'statistics',
    descriptor: {scope: 'frontier', reduction: 'argmin'},
    variables: {
      positions: 'coordinate',
      ordering: 'rank',
      coreDistances: 'measure',
      reachability: 'measure',
      predecessors: 'category',
      clusterIntervals: 'rank',
      clusterSummary: 'category',
      labels: 'category',
      status: 'category'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'point', label: 'event', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'minimumSamples', stage: 'neighbourhood', sweepable: true},
      {name: 'maximumDistance', stage: 'neighbourhood', sweepable: true},
      {name: 'extractionDistance', stage: 'operator', sweepable: true},
      {
        name: 'extractionMethod',
        stage: 'operator',
        alternatives: ['dbscan', 'xi'],
        sweepable: true
      },
      {name: 'xi', stage: 'operator', sweepable: true},
      {name: 'minimumClusterSize', stage: 'operator', sweepable: true},
      {
        name: 'neighborIndex',
        stage: 'neighbourhood',
        alternatives: ['auto', 'brute-force', 'grid'],
        sweepable: false
      },
      {
        name: 'orderingMethod',
        stage: 'operator',
        alternatives: ['sequential', 'parallel'],
        sweepable: false
      },
      {name: 'frontierBatchSize', stage: 'operator', sweepable: false},
      {name: 'maximumIterations', stage: 'operator', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Sequential cooperative or parallel indexed minimum-reachability frontier ordering and DBSCAN/xi hierarchy extraction are procedural; core distances use self-inclusive minimumSamples.'
    },
    metadata: {
      coordinateModel: 'projected',
      extraction: 'dbscan-or-xi',
      exact: true,
      maximumEvents: 65536,
      maximumBruteForceEvents: 4096
    }
  },
  GPUSpatialBalancedSample: {
    family: 'statistics',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      frameIds: 'category',
      inclusionProbabilities: 'probability',
      designWeights: 'rate',
      status: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'point', label: 'finite-frame unit', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'seed', stage: 'operator', sweepable: true},
      {name: 'replicate', stage: 'operator', sweepable: true},
      {name: 'allocations', stage: 'support', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Randomized hierarchical traversal plus integer systematic selection implements exact finite-frame first-order inclusion probabilities, with inverse-probability design weights.'
    },
    metadata: {
      design: 'hierarchical-circular-systematic',
      frame: 'finite-points',
      grtsParity: false,
      unequalProbabilities: true
    }
  },
  GPUContinuousSpatialSample: {
    family: 'statistics',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      designWeights: 'rate',
      provenance: 'category',
      status: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'point', label: 'continuous-frame survey site', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'spacing', stage: 'operator', sweepable: true},
      {name: 'seed', stage: 'operator', sweepable: true},
      {name: 'waves', stage: 'operator', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Uniform lattice phase gives first-order intensity 1/spacing on arc-length frames or 1/spacing² on triangle-union area frames. Fixed legacy exclusion preserves intensity on the eligible frame; independently randomized replacement waves have explicit provenance.'
    },
    metadata: {
      design: 'random-shift-lattice',
      continuousFrames: true,
      grtsParity: false,
      replacement: 'independent-complete-waves',
      phasePrecision: '24-bit',
      maximumCandidates: 1048576
    }
  }
};
