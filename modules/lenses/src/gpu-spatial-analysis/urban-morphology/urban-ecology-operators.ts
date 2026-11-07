// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../computation-pack/types';

/** Resident urban/ecology recipe discovery, with approximation and solver boundaries preserved. */
export const URBAN_ECOLOGY_OPERATORS: SpatialFamilyOperators = {
  addBuildingTessellationRecipe: {
    family: 'recipes',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {
      positions: 'coordinate',
      sourceIds: 'category',
      featureOffsets: 'category',
      polygonOffsets: 'category',
      ringOffsets: 'category',
      segments: 'coordinate',
      segmentBuildingIds: 'category',
      enclosure: 'coordinate',
      samples: 'coordinate',
      owners: 'category',
      fragments: 'coordinate',
      fragmentCounts: 'count',
      complete: 'category'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [
      {kind: 'building', label: 'building cell', indexSpace: '1d'},
      {kind: 'site', label: 'boundary sample', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {name: 'sampleSpacing', stage: 'representation', sweepable: true},
      {name: 'vertexTolerance', stage: 'operator', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Building boundary sampling feeds half-plane-clipped sample cells in a convex bound with optional nonconvex enclosure overlay; GPUPolygonOverlay dissolves same-building fragments on an explicit snap lattice. Published geometry carries stable building IDs and independent sampling/overlay completeness.'
    },
    metadata: {
      coordinateModel: 'projected',
      enclosure: 'convex-or-simple-overlay',
      maximumSamples: 128,
      buildingShrink: 'explicit-preprocessing'
    }
  },
  addStreetProfilesRecipe: {
    family: 'recipes',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    variables: {
      positions: 'coordinate',
      normals: 'direction',
      profiles: 'measure',
      buildings: 'category',
      validity: 'category',
      canyon: 'rate',
      frontage: 'measure',
      lengths: 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {kind: 'sample', label: 'street cross-section', indexSpace: '1d'},
      {kind: 'building', label: 'frontage owner', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {name: 'maximumDistance', stage: 'neighbourhood', sweepable: true},
      {name: 'sampleSpacing', stage: 'support', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Opposing normalized street-normal rays select the closest closed building segment. Observed canyon width requires both sides; range-censored widths and missing heights remain separate. Sample frontage sums interval lengths by side and building; optional straight-corridor facade visibility splits segment events and measures exposed arc length.'
    },
    metadata: {
      coordinateModel: 'projected',
      missingSide: 'range-censored',
      heightUnits: 'match-planar-units'
    }
  },
  addBuildingAdjacencyRecipe: {
    family: 'recipes',
    descriptor: {scope: 'pairwise', reduction: 'sum'},
    variables: {
      orientations: 'direction',
      morphology: 'measure',
      components: 'category',
      lineBuildingIds: 'category',
      pairs: 'measure',
      lengths: 'measure',
      complete: 'category'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'pair', label: 'building pair', indexSpace: '2d'}],
    guards: ['coordinate-model'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Consumes existing shared-path runs and sums positive wall lengths by stable building pair. Point contacts do not become wall adjacency; optional explicit Queen/custom support defines structure components. Axial differences wrap modulo pi; cardinal-deviation degrees and neighbor alignment reuse supplied shape orientations. Optional positive-winding component overlay measures exterior-only union perimeter, excluding courtyard holes and exposing every overlay status.'
    },
    metadata: {
      coordinateModel: 'projected',
      orientationPeriod: Math.PI,
      duplicatePairPolicy: 'caller-supplies-once'
    }
  },
  addIsovistsRecipe: {
    family: 'recipes',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    variables: {
      positions: 'coordinate',
      endpoints: 'coordinate',
      distances: 'measure',
      segments: 'category',
      validity: 'category',
      measures: 'measure',
      angles: 'direction',
      viewpointClassifications: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {kind: 'viewpoint', label: 'free-space viewpoint', indexSpace: '1d'},
      {kind: 'ray', label: 'angular sample', indexSpace: '2d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {name: 'rayCount', stage: 'support', sweepable: true},
      {name: 'maximumDistance', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Uniform or sorted vertex-event angular rays stop at the closest closed obstacle or finite range boundary. Adjacent endpoints define an approximate polygon whose area/perimeter are measured separately; boundary viewpoints are invalid; optional even-odd/nonzero filled rings classify interior/exterior free space and holes. Angular resolution remains explicit.'
    },
    metadata: {
      coordinateModel: 'projected',
      approximation: 'uniform-or-vertex-event-angular-sampling',
      viewpointOnBoundary: 'invalid'
    }
  },
  addVisibilityGraphRecipe: {
    family: 'recipes',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'sum'},
    variables: {
      positions: 'coordinate',
      adjacency: 'category',
      edges: 'category',
      metrics: 'score',
      choice: 'score',
      integration: 'score',
      normalizedChoice: 'score',
      extendedIntegration: 'score',
      viewpointClassifications: 'category',
      status: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {kind: 'viewpoint', label: 'visibility vertex', indexSpace: '1d'},
      {kind: 'edge', label: 'visible pair', indexSpace: '2d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {name: 'maximumDistance', stage: 'neighbourhood', sweepable: true},
      {name: 'edgeCapacity', stage: 'operator', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Closed-obstacle pair visibility builds a complete dense graph and a capacity-bounded edge list. BFS computes reachable-hop closeness and Brandes dependencies compute raw and normalized undirected choice; reachable mean depth produces relative asymmetry, real relative asymmetry and Hillier-Hanson, pyramid P-value, Teklenburg and Penn normalization using actual maximum shortest-path depth; edge-list overflow is distinct from dense-metric validity.'
    },
    metadata: {
      coordinateModel: 'projected',
      maximumViewpoints: 256,
      integration: 'reachable-hop-closeness-and-Hillier-Hanson',
      choice: 'raw-and-normalized-undirected-betweenness'
    }
  },
  addLandscapeMetricsRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', scopeSecondary: 'global', reduction: 'sum'},
    variables: {
      patches: 'category',
      patchClasses: 'category',
      patchMetrics: 'measure',
      classMetrics: 'measure',
      classAdjacency: 'count',
      aggregation: 'rate',
      landscapeAggregation: 'rate',
      landscapeCohesion: 'rate',
      landscape: 'score'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator'],
    datumKinds: [
      {kind: 'patch', label: 'connected patch', indexSpace: '1d'},
      {kind: 'class', label: 'landscape class', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      {name: 'connectivity', stage: 'representation', alternatives: ['4', '8'], sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Optional GPU seam reconciliation merges globally unique labels on resident halo-free support. Explicit boundary and background policies feed physical area/perimeter, orthogonal directed adjacency and class reductions. Mesh uses sum of squared patch areas over landscape area; explicit area-weighted or joint-adjacency contagion, compact-square aggregation and cohesion formulas retain background and undefined-denominator conventions.'
    },
    metadata: {
      coordinateModel: 'projected',
      classAdjacency: 'four-neighbor',
      tiledInput: 'resident-global-ID-reconciliation'
    }
  },
  addPatchProximityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    variables: {
      patches: 'category',
      patchClasses: 'category',
      core: 'category',
      distances: 'measure',
      metrics: 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'patch', label: 'connected patch', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {name: 'edgeDepth', stage: 'support', sweepable: true},
      {name: 'searchRadius', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Physical depth or four-neighbor raster erosion determines core membership; open/edge domain policy is explicit. separable rectangle-boundary or cell-center distance fields exclude self and restricts nearest/proximity summaries to same-class patches. Proximity sums neighboring patch area over squared separation inside the supplied radius.'
    },
    metadata: {
      coordinateModel: 'projected',
      distanceMetric: 'euclidean-cell-boundary',
      tiledInput: 'resident-global-ID-reconciliation'
    }
  },
  addResistanceConnectivityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'solve'},
    variables: {
      edges: 'category',
      conductances: 'rate',
      potentials: 'score',
      currents: 'score',
      diagnostics: 'score',
      status: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [
      {kind: 'vertex', label: 'conductance vertex', indexSpace: '1d'},
      {kind: 'edge', label: 'resistor edge', indexSpace: '1d'}
    ],
    guards: [],
    choices: [
      {name: 'tolerance', stage: 'operator', sweepable: true},
      {name: 'maximumIterations', stage: 'operator', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Unit injection on the sink-connected component is solved with a grounded conductance Laplacian and bounded conjugate gradients. Independently recomputed true current residual gates effective resistance and edge-current publication; disconnection and iteration exhaustion remain explicit.'
    },
    metadata: {
      maximumVertices: 256,
      injection: 1,
      disconnectedPotential: 'NaN',
      convergence: 'true-maximum-current-residual'
    }
  },
  addSegregationDecompositionRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      populations: 'count',
      indices: 'rate',
      validity: 'category',
      decomposition: 'score'
    },
    stages: ['support', 'operator'],
    datumKinds: [
      {kind: 'unit', label: 'common-support population unit', indexSpace: '1d'},
      {kind: 'scenario', label: 'explicit counterfactual', indexSpace: '1d'}
    ],
    guards: [],
    choices: [{name: 'counterfactualMethod', stage: 'support', sweepable: false}],
    formula: {
      formula: 'none',
      reason:
        'Four explicitly aligned focal/complement population tables produce dissimilarity indices. The two-factor Shapley average separates composition and spatial changes and publishes their conservation residual; optional paired-label randomization records its exchangeability assumption and replay stream. Counterfactual construction is a separate explicit recipe.'
    },
    metadata: {
      statistic: 'dissimilarity',
      decomposition: 'two-factor-shapley',
      supportAlignment: 'caller-explicit'
    }
  },
  addSparseResistanceConnectivityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      potentials: 'score',
      currents: 'score',
      diagnostics: 'score',
      status: 'category',
      components: 'category',
      residual: 'score'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'unit', label: 'analysis unit', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Sparse Jacobi-preconditioned resistance with a true-current certificate. Construction, status and numerical outputs are recorded in the caller graph; no implicit host readback.'
    },
    metadata: {execution: 'graph-native', reference: 'explicit-domain-contract'}
  },
  addRasterResistanceConnectivityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      resistance: 'measure',
      conductances: 'measure',
      potentials: 'score',
      currents: 'score',
      diagnostics: 'score',
      status: 'category',
      currentDensity: 'score'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'unit', label: 'analysis unit', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Raster resistance converted to physical neighbor conductance. Construction, status and numerical outputs are recorded in the caller graph; no implicit host readback.'
    },
    metadata: {execution: 'graph-native', reference: 'explicit-domain-contract'}
  },
  addSegregationCounterfactualRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      baseline: 'count',
      final: 'count',
      compositionOnly: 'count',
      spatialOnly: 'count',
      indices: 'rate',
      validity: 'category',
      decomposition: 'score'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'unit', label: 'analysis unit', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Empirical-quantile population counterfactuals and Shapley components. Construction, status and numerical outputs are recorded in the caller graph; no implicit host readback.'
    },
    metadata: {execution: 'graph-native', reference: 'explicit-domain-contract'}
  },
  addPopulationSupportAlignmentRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      populations: 'count',
      sourceIds: 'category',
      targetIds: 'category',
      validity: 'category',
      status: 'category'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'unit', label: 'analysis unit', indexSpace: '1d'}],
    guards: [],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Explicit population support identifiers reordered with completeness evidence. Construction, status and numerical outputs are recorded in the caller graph; no implicit host readback.'
    },
    metadata: {execution: 'graph-native', reference: 'explicit-domain-contract'}
  }
};
