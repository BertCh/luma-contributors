// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/** Expansion contracts; structural and iterative operators are explained through their real graph. */
const expansionOperators: SpatialFamilyOperators = {
  GPULocalSpatialHeteroscedasticity: {
    family: 'statistics',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {statistic: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'LOSH divides a weighted neighborhood residual moment by the study-wide residual moment; the graph exposes normalization and invalid-row status.'
    }
  },
  GPULOSHResidualPermutationTest: {
    family: 'statistics',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {statistic: 'score', pseudoPValues: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Conditional randomization tests the fixed normalized LOSH residual field with an explicit tail; residuals are not refitted within permutations.'
    }
  },
  GPUMultivariateLocalGeary: {
    family: 'statistics',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {statistic: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Complete-case, variance-standardized multivariate squared differences are averaged over each weighted neighborhood.'
    }
  },
  GPUGeneralizedGeographicallyWeightedRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {coefficients: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Family-specific local IRLS requires iterative weighted normal-equation solves and convergence status; one scalar formula would omit the estimator contract.'
    }
  },
  GPUMultiscaleGeographicallyWeightedRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {coefficients: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Additive multiscale backfitting alternates coefficient updates with explicitly selected bandwidths and bounded convergence checks.'
    }
  },
  GPUSpatialPanelRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {table: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Pooled or unit-effects spatial-lag two-stage least squares constructs period-block weights and delegates explicit instrument and regression stages; unit effects first apply the within transformation.'
    }
  },
  GPUSpatialRegimeRegression: {
    family: 'regression',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {coefficients: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Separate fully varying ordinary least-squares models are solved per caller-defined regime, retaining empty and singular group status.'
    }
  },
  GPUConcaveHull: {
    family: 'geometry',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {triangleValidity: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'An alpha complex selects Delaunay triangles by maximum circumradius and derives the exposed boundary edges; graph and validity are the appropriate structural explanation.'
    }
  },
  GPUGroupConcaveHull: {
    family: 'geometry',
    descriptor: {scope: 'zonal', reduction: 'select'},
    variables: {triangleValidity: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Independent alpha complexes preserve original point IDs and empty group slots; boundary and triangle capacities remain explicit.'
    }
  },
  GPUConstrainedDelaunayTriangulation: {
    family: 'geometry',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {indices: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Polygon triangulation preserves ring constraints and performs legal Lawson edge flips, retaining validity and convergence diagnostics.'
    }
  },
  GPUH3CellIdentities: {
    family: 'cells',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {output: 'category'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Integer H3 topology derives six canonical directed-edge or vertex keys, including deleted pentagon slots; numerical interpolation is not involved.'
    }
  },
  GPUBoundedFacilityAllocation: {
    family: 'accessibility',
    descriptor: {scope: 'global', reduction: 'argmin'},
    variables: {objective: 'measure'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Bounded facility-subset enumeration separates exact p-center certification from demand-order greedy capacitated allocation.'
    }
  },
  addZonalDistributionRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'scan-sort'},
    variables: {medians: 'rate'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Composes finite-value filtering, dense zone grouping, ordered quantiles and deterministic modes; a graph explains the separate reductions.'
    }
  },
  addRasterZonalDistributionRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'scan-sort'},
    variables: {medians: 'rate'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Native nodata is filtered before scale/offset calibration, then the zonal distribution stages run on calibrated values.'
    }
  },
  addExposureByZoneRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {exposedShare: 'rate'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Exposure separates total population, hazard-observed population and threshold-exposed population, with explicit missing-support diagnostics.'
    }
  },
  addRegionalRasterChangeRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'weighted-mean'},
    variables: {meanChanges: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Aligned slice change detection feeds zone summaries on common support; cell areas, missing support and threshold validity remain separate outputs.'
    }
  },
  addSurfacePreparationRecipe: {
    family: 'recipes',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {values: 'rate'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Bounded filling and optional rank filtering preserve original support provenance and report filled, filtered and unresolved cells separately.'
    }
  },
  addInterpolationDiagnosticsRecipe: {
    family: 'recipes',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {residuals: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Leave-one-out kriging produces residuals and variance-aware validation summaries with explicit invalid and standardized-count diagnostics.'
    }
  },
  addDrainageNetworkRecipe: {
    family: 'recipes',
    descriptor: {scope: 'frontier', reduction: 'scan-sort'},
    variables: {distance: 'measure'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'D8 routing, stream extraction, downstream distance and reach assembly form a multi-stage drainage graph with convergence and capacity status.'
    }
  },
  addLocalVariabilityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {losh: 'score'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Shared weights feed descriptive LOSH and analytic local Moran inference, preserving separate inference families and row validity.'
    }
  },
  addCellFlowTopologyRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {outgoingTotals: 'count'},
    stages: ['operator'],
    datumKinds: [{kind: 'row', label: 'output row', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Original OD pairs feed conserved cell totals and a separate immediate-neighbor edge aggregation with exact H3 identities and boundaries.'
    }
  }
};

/** Curated scalar definitions; masks and status remain visible in the graph lens. */
export const EXPANSION_OPERATORS: SpatialFamilyOperators = {
  ...expansionOperators,
  GPULocalSpatialHeteroscedasticity: {
    ...expansionOperators['GPULocalSpatialHeteroscedasticity'],
    formula: {
      id: 'local-spatial-heteroscedasticity',
      derivation: 'curated',
      tex: String.raw`H_i=\frac{\sum_j w_{ij}e_j^a/\sum_jw_{ij}}{\sum_{k\in V}e_k^a/|V|},\quad e_i=|x_i-\sum_jw_{ij}x_j/\sum_jw_{ij}|`,
      ascii:
        'H_i = weightedMean_j(e_j^a) / mean_k(e_k^a), e_i = abs(x_i - weightedMean_j(x_j)), a in {1,2}. V contains finite residual rows; positive usable neighbors are renormalized. Undefined means retain status.',
      symbols: [
        {id: 'statistic', role: 'output', binding: 'wgsl:result'},
        {id: 'values', role: 'input', binding: 'wgsl:values'},
        {id: 'weights', role: 'input', binding: 'wgsl:weights'}
      ]
    }
  },
  GPUMultivariateLocalGeary: {
    ...expansionOperators['GPUMultivariateLocalGeary'],
    formula: {
      id: 'multivariate-local-geary',
      derivation: 'curated',
      tex: String.raw`c_i=\frac{1}{p\sum_jw_{ij}}\sum_jw_{ij}\sum_{q=1}^p\frac{(x_{iq}-x_{jq})^2}{s_q^2}`,
      ascii:
        'c_i = sum_j w_ij sum_q (x_iq - x_jq)^2 / s_q^2 / (p sum_j w_ij). Complete-case population variances use included rows. Islands and zero variance are undefined.',
      symbols: [
        {id: 'statistic', role: 'output', binding: 'wgsl:result'},
        {id: 'values', role: 'input', binding: 'wgsl:values'},
        {id: 'weights', role: 'input', binding: 'wgsl:weights'}
      ]
    }
  }
};
