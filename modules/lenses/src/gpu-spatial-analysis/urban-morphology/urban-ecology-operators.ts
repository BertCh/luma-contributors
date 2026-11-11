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
        'Building boundary sampling feeds half-plane-clipped sample cells in a convex bound with optional nonconvex enclosure overlay; GPUPolygonOverlay dissolves same-building fragments on an explicit snap lattice. The geometry of each building is its own overlay output, indexed by building row, with independent sampling/overlay completeness.'
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
  addSparseResistanceConnectivityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'solve'},
    variables: {
      edges: 'category',
      conductances: 'rate',
      potentials: 'score',
      currents: 'score',
      vertexCurrents: 'score',
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
      {name: 'maximumIterations', stage: 'operator', sweepable: true},
      {name: 'connectivityIterations', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'gpu-spatial-analysis.sparse-resistance-connectivity-recipe',
      derivation: 'curated',
      tex: String.raw`\sum_{j}g_{ij}\left(\term{potentials}{v_i}-v_j\right)=[i=s]-[i=t],\quad v_t=0,\quad R_{\mathrm{eff}}=v_s,\quad \term{vertexCurrents}{I_i}=\tfrac12\Big(\sum_j\left|g_{ij}(v_i-v_j)\right|+[i\in\{s,t\}]\Big)`,
      ascii:
        'Unit current enters at the source s and leaves at the sink t, held at potential 0. Potentials solve the grounded conductance Laplacian by Jacobi-preconditioned conjugate gradients; the effective resistance is the source potential and the node current is half the summed absolute branch currents plus the injection.',
      symbols: [
        {id: 'potentials', role: 'output', label: 'v_i', binding: 'wgsl:potentials'},
        {id: 'vertexCurrents', role: 'output', label: 'I_i', binding: 'wgsl:vertexCurrents'}
      ]
    },
    metadata: {
      maximumVertices: 1048576,
      maximumEdges: 2097152,
      injection: 1,
      solver: 'grounded-jacobi-conjugate-gradient',
      disconnectedPotential: 'NaN',
      convergence: 'true-maximum-current-residual'
    }
  },
  addRasterResistanceConnectivityRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', reduction: 'solve'},
    variables: {
      resistance: 'rate',
      validity: 'category',
      edges: 'category',
      conductances: 'rate',
      potentials: 'score',
      currents: 'score',
      vertexCurrents: 'score',
      diagnostics: 'score',
      solverStatus: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'raster cell', indexSpace: '2d'}],
    guards: [],
    choices: [
      {name: 'neighborhood', stage: 'neighbourhood', sweepable: false},
      {name: 'cellSizeMode', stage: 'support', sweepable: false},
      {name: 'tolerance', stage: 'operator', sweepable: true},
      {name: 'maximumIterations', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'gpu-spatial-analysis.raster-resistance-connectivity-recipe',
      derivation: 'curated',
      tex: String.raw`\term{conductances}{g_{ab}}=\frac{2}{\term{resistance}{R_a}+R_b}\cdot\frac{1}{\ell_{ab}},\quad \sum_{b\sim a}g_{ab}\left(\term{potentials}{v_a}-v_b\right)=[a=s]-[a=t],\quad v_t=0`,
      ascii:
        'Neighbouring cells a and b with data (4 or 8 neighbours) are joined by the conductance 2 / (R(a) + R(b)) / edgeLength, the edge length being the ground distance between cell centres under the grid and cellSizeMode. No-data cells are removed; zero or negative resistance is invalid. Unit current from source cell s to sink cell t is solved by the sparse resistance recipe.',
      symbols: [
        {id: 'resistance', role: 'input', label: 'R_a', binding: 'wgsl:resistance'},
        {id: 'conductances', role: 'intermediate', label: 'g_ab', binding: 'wgsl:conductances'},
        {id: 'potentials', role: 'output', label: 'v_a', binding: 'wgsl:potentials'}
      ]
    },
    metadata: {
      maximumCells: 1048576,
      maximumEdgeSlots: 2097152,
      conductance: '2 / (R(a) + R(b)) / edgeLength',
      noData: 'removed',
      nonPositiveResistance: 'invalid',
      solver: 'addSparseResistanceConnectivityRecipe'
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
