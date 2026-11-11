// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../computation-pack/types';

/** Domain-owned bounded statistical contracts; research exclusions remain visible. @internal */
export const STATISTICS_COVERAGE_OPERATORS: SpatialFamilyOperators = {
  GPUSpatialPanelLagMaximumLikelihood: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Panel estimate', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Exact entity log determinants and alternating quasi-demeaning profile spatial-lag random effects; conditional information and a separately fitted unit-effects Hausman comparison expose convergence and degeneracy.'
    },
    metadata: {
      backlog: 'ST03.random',
      supported:
        'balanced random/unit spatial-lag ML; conditional coefficient covariance; computed Hausman',
      excludes: 'time effects, clustered covariance, incomplete panels'
    }
  },
  GPUInstrumentalVariablesCovariance: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Centered/scaled instrument projection forms HC0, HC1 or spatial-HAC influence sandwiches; full covariance and invalid-kernel status are explicit.'
    },
    metadata: {
      backlog: 'ST01',
      supported: 'HC0, HC1, symmetric unit-diagonal nonnegative PSD HAC kernel',
      excludes: 'HC2/HC3, clustered covariance, automatic PSD certification'
    }
  },
  GPURestrictedRegimeRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Common and varying regime coefficients use nested dense solves; full covariance and the joint Chow F statistic retain singular-regime status.'
    },
    metadata: {
      backlog: 'ST02',
      supported:
        'common/varying intercept and slopes; homoskedastic nested Chow F and coefficient/joint Wald chi-square tails',
      excludes: 'spatially robust restrictions'
    }
  },
  GPUSpatialPanelErrorMaximumLikelihood: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Unit demeaning and a time-block spatial graph profile the fixed-effects error likelihood with explicit trace-series error and bounded parameter search.'
    },
    metadata: {
      backlog: 'ST03',
      supported: 'balanced unit FE spatial-error ML; pooled LM error and robust LM lag/error',
      excludes: 'random effects, time effects, Hausman and clustered inference'
    }
  },
  GPUSeeminglyUnrelatedRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'First-stage OLS residuals estimate the joint equation covariance before one block feasible-GLS solve and a full covariance transform.'
    },
    metadata: {
      backlog: 'ST04',
      supported:
        'one-step FGLS and Kelejian–Prucha spatial-error SUR; explicit aligned complete-case contracts',
      excludes: 'iterated SUR'
    }
  },
  GPUProbitRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Centered/scaled Fisher scoring with likelihood backtracking returns model-based errors, average marginal effects and explicit separation/exhaustion.'
    },
    metadata: {
      backlog: 'ST05',
      supported:
        'binary probit, expected-information uncertainty and three model-adjusted spatial residual tests',
      excludes: 'latent spatial probit'
    }
  },
  GPULocalCollinearity: {
    family: 'regression',
    descriptor: {scope: 'focal', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Each supplied local estimation neighborhood forms a weighted predictor correlation matrix and its inverse diagonal and eigenvalue ratio.'
    },
    metadata: {
      backlog: 'ST06',
      supported:
        'local VIF, centered condition index, intercept-inclusive design condition and variance decomposition',
      excludes: 'multiscale backfitting uncertainty'
    }
  },
  GPUMultivariateLocalGearyPermutationTest: {
    family: 'statistics',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Common masks and population standardization precede whole-vector conditional sampling without replacement and conservative equal-tail classification.'
    },
    metadata: {
      backlog: 'ST07',
      supported: 'joint local randomization, replay, optional Bonferroni family',
      excludes: 'independent-column permutations, exact upstream class-label parity'
    }
  },
  GPURankMarkov: {
    family: 'dynamics',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Stable descending ranks or their geographic occupants feed exact transition counts and existing Markov diagnostics; missing panels use common complete cases.'
    },
    metadata: {
      backlog: 'ST08',
      supported: 'full-rank and geographic-rank transitions, mean absolute rank movement',
      excludes: 'fractional ties, missing-period interpolation, unrestricted state counts'
    }
  },
  GPUFuzzyContiguity: {
    family: 'weights',
    descriptor: {scope: 'pairwise', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Filled-polygon separation uses segment distances and multipart/hole containment, then deterministic bounded CSR construction with tolerance-edge provenance.'
    },
    metadata: {
      backlog: 'ST09',
      supported: 'planar polygon separation tolerance with empty/malformed status',
      excludes: 'geographic geometry, topology repair, GEOS robust-predicate parity'
    }
  },
  GPUIslandAttachment: {
    family: 'weights',
    descriptor: {scope: 'local', reduction: 'select'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Original empty rows select the nearest finite representative; deterministic edge union retains original weights and added-edge provenance.'
    },
    metadata: {
      backlog: 'ST09',
      supported: 'simultaneous directed or reciprocal nearest attachment',
      excludes: 'automatic row normalization, inferred representative coordinates'
    }
  },
  GPUSpatialCrossValidation: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Seeded block assignment (each block ranked by a Philox key, fold = rank modulo foldCount, so folds differ by at most one block and none is empty when blocks >= folds) and exact pair-distance exclusions produce distinct test/training masks with empty-fold status.'
    },
    metadata: {
      backlog: 'ST10',
      supported:
        'planar block/group/environmental folds, held-out buffers and bounded NNDM/kNNDM distance matching',
      excludes: 'R RNG identity, automatic polygon blocks, response-balanced optimization'
    }
  },
  GPUSpatialCrossValidatedOLS: {
    family: 'recipes',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Independent OLS fits estimate training-only centered designs, then predict held-out rows and report scoring coverage and error metrics.'
    },
    metadata: {
      backlog: 'ST10',
      supported: 'repeated OLS fits with training-only preprocessing and held-out metrics',
      excludes: 'nested bandwidth/variogram selection for other models'
    }
  },
  GPUSpatialInteraction: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {status: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'Statistical output', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'An explicit coefficient grid or bounded continuous search profiles Poisson flow means; bounded proportional fitting enforces selected margins and exposes candidate convergence/deviance.'
    },
    metadata: {
      backlog: 'ST11',
      supported:
        'unconstrained, origin, destination and doubly constrained one-covariate grid or bounded continuous calibration',
      excludes: 'coefficient uncertainty, arbitrary count distributions'
    }
  }
};
