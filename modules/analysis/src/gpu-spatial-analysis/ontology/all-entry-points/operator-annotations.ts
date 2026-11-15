// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisLimit, GPUAnalysisTier} from '../spatial-analysis-ontology';

/** Tier, limits and role of an operator, by export name. Operators not listed are `'scale'`. */
export type OperatorAnnotation = {
  tier?: GPUAnalysisTier;
  limits?: readonly GPUAnalysisLimit[];
  limitation?: string;
  role?: 'operator' | 'helper';
};

/**
 * Limits and tiers confirmed against the source on 2026-11-09 (each `source` is the file that
 * enforces the maximum). A `reference` operator is a single-invocation, exhaustive or small-dense
 * design with a hard maximum size; every other operator is `scale`. Several of these contributors
 * are being reworked, so a limit that disappears from its source must be removed here as well.
 */
export const OPERATOR_ANNOTATIONS: Readonly<Record<string, OperatorAnnotation>> = {
  getHealpixCellBoundary: {
    role: 'helper',
    limitation:
      'Sampled spherical display geometry; longitudes can extend beyond +/-180. Not an ellipsoidal or OGC ZIRS boundary.',
    limits: [
      {
        name: 'order',
        maximum: 29,
        reason: 'The nested HEALPix 64-bit key layout.',
        source: 'modules/analysis/src/gpu-spatial-analysis/healpix-cells/healpix-cell-boundary.ts'
      },
      {
        name: 'edgeSegments',
        maximum: 64,
        reason: 'Bounds the display ring to 257 positions.',
        source: 'modules/analysis/src/gpu-spatial-analysis/healpix-cells/healpix-cell-boundary.ts'
      }
    ]
  },
  GPUQuantileLISA: {
    tier: 'scale',
    limits: [
      {
        name: 'values.length',
        maximum: 16777215,
        reason:
          'Quantile ranks and conditional permutation positions require exact float32 integers.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa.ts'
      },
      {
        name: 'variable count',
        maximum: 64,
        reason: 'Bound compilation cost of two resident quantile passes per variable.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa.ts'
      },
      {
        name: 'maximumNeighbors',
        maximum: 64,
        reason: 'Shared conditional permutation per-thread scratch capacity.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test.ts'
      },
      {
        name: 'maximumPermutations',
        maximum: 1048576,
        reason: 'Shared conditional permutation loop bound.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels.ts'
      }
    ],
    limitation:
      'Per-variable type-5 membership intersection, with common-mask and per-variable missing-value omission. Empty intersections remain empty; no bivariate no-colocation fallback. Legacy directed p-values can be anti-conservative. The eight-column inference scratch matrix must fit one device storage binding.'
  },
  GPUCoregionalizationFit: {
    tier: 'reference',
    limitation:
      'Complete lag matrices with shared pair weights; no range search. Goulard & Voltz (1992), doi:10.1007/BF00893750. Prefer CPUCoregionalizationFitOperation.',
    limits: [
      {
        name: 'variableCount',
        maximum: 4,
        reason: 'Local co-kriging square loading layout.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-parameters.ts'
      },
      {
        name: 'structureCount',
        maximum: 4,
        reason: 'Local co-kriging structure layout.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-parameters.ts'
      },
      {
        name: 'distances.length',
        maximum: 256,
        reason: 'Reference-tier serial sufficient statistics.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 1024,
        reason: 'Two command nodes per gated sweep.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-parameters.ts'
      }
    ]
  },
  GPUIsolines: {
    limits: [
      {
        name: 'domain samples',
        maximum: 2147483647,
        reason: 'Two raster-edge families must fit in uint32 edge identifiers.',
        source: 'modules/analysis/src/raster-analysis/isolines/isolines-tile.ts'
      },
      {
        name: 'owned cells * levels.length * 2',
        maximum: 4294967295,
        reason: 'Compact segment counts and scan offsets use uint32.',
        source: 'modules/analysis/src/raster-analysis/isolines/isolines-tile.ts'
      }
    ],
    limitation:
      'Tile ownership rectangles must partition domain cells; stitched polylines are tile-local fragments. Placement uses a shared extent with row zero at minY, not the grid affine.'
  },
  GPUHierarchicalDensityClustering: {
    tier: 'scale',
    limits: [
      {
        name: 'positions.length',
        maximum: 1048576,
        reason: 'Bounds MST, dendrogram and condensed-tree scratch.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts'
      },
      {
        name: 'minimumSamples',
        maximum: 32,
        reason: 'Bounds the indexed per-query top-k list.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts'
      }
    ],
    limitation:
      'Planar HDBSCAN or scaled space-time HDBSCAN; scikit-learn Euclidean EOM defaults. Equal-distance MST ties use input row order. The spatial BVH lower bound does not prune by time.'
  },
  GPUGeodesicPairs: {
    limits: [
      {
        name: 'maximumIterations',
        maximum: 200,
        reason: 'Bounded per-pair float32 Vincenty work.',
        source: 'modules/analysis/src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts'
      }
    ],
    limitation:
      'WGS84 inverse within 0.02 radians of an antipode uses a sphere fallback with converged=0 and approximation=1. Spherical midpoints within 1e-6 radians of an antipode are NaN. See modules/analysis/docs/geodesic-precision.md for measured arithmetic envelopes and pending GPU certification.'
  },
  GPUGeodesicDestination: {
    limits: [
      {
        name: 'maximumIterations',
        maximum: 200,
        reason: 'Bounded per-row float32 Vincenty work.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination.ts'
      }
    ],
    limitation:
      'Float32 coordinates and bounded Vincenty iteration; unconverged results are published with iterationLimitReached=1. Nonzero-distance arrivals with abs(cos(latitude)) <= 0.03 report approximation=1 because azimuth is ill-conditioned. WGS84 and sphere envelopes are documented in modules/analysis/docs/geodesic-precision.md; arbitrary custom ellipsoids are outside those measurements.'
  },
  ...Object.fromEntries(
    [
      'GPUGeographicallyTemporallyWeightedRegression',
      'GPUMultiscaleGeographicallyTemporallyWeightedRegression'
    ].map(exportName => [
      exportName,
      {
        tier: 'reference' as const,
        limits: [
          {
            name: 'rows',
            maximum: 32768,
            reason: 'Exhaustive local weighted least squares and serial bounded summaries.',
            source:
              'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression.ts'
          },
          {
            name: 'predictorCount',
            maximum: 11,
            reason: 'Private dense Cholesky systems.',
            source:
              'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression.ts'
          },
          {
            name: 'adaptive bandwidth',
            maximum: 128,
            reason: 'Private nearest-neighbour distance list, including self.',
            source:
              'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gtwr-bandwidth-search.ts'
          },
          {
            name: 'bandwidthSearch.maximumIterations',
            maximum: 32,
            reason:
              'Nested golden-section searches unroll into commands; the 32768-command budget additionally limits combinations.',
            source:
              'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gtwr-bandwidth-search.ts'
          }
        ],
        limitation:
          'Planar native metric; exhaustive space-time distances. Golden-section profile search assumes unimodality, tests endpoints and reports interval convergence; adaptive neighbour ties can make the objective nonsmooth.'
      }
    ])
  ),
  GPUForestInference: {
    tier: 'scale',
    limitation:
      'Numerical splits and constant leaves only; caller-supplied models, float32 features and scores. No DART, categorical splits, multi-target trees or fitting.',
    limits: [
      ...[
        ['treeCount', 1024, 'Bounded traversal per row.'],
        ['maximumDepth', 64, 'Bounded sparse tree walk.'],
        ['nodeCount', 1048576, 'Sparse node storage is at most 24 MiB.'],
        ['featureCount', 256, 'Bounded feature index.'],
        ['outputCount', 32, 'Per-invocation score array.'],
        [
          'rowCount',
          16777216,
          'Bounded uint32 indexing; binding checks may require smaller batches.'
        ],
        ['rasterStack.bands.length', 16, 'One preparation dispatch per band.'],
        [
          'worst-case work',
          5000000000,
          'Split batches to bound traversal and score accumulation work.'
        ]
      ].map(([name, maximum, reason]) => ({
        name: String(name),
        maximum: Number(maximum),
        reason: String(reason),
        source: 'modules/analysis/src/dataframe-analysis/forest-inference/gpu-forest-inference.ts'
      }))
    ]
  },
  GPUSpaceTimeKriging: {
    tier: 'reference',
    limits: [
      {
        name: 'positions.length',
        maximum: 1048576,
        reason: 'Every query scans every sample in one invocation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts'
      }
    ]
  },
  GPURasterFalseDiscoveryRateMask: {
    limits: [
      {
        name: 'cellCount',
        maximum: 16777215,
        reason: 'Ranks and the family size are exact float32 integers below 2^24.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment.ts'
      },
      {
        name: 'histogram bins',
        maximum: 65536,
        reason: 'The published histogram clears and bins in one bounded dispatch.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask.ts'
      }
    ]
  },
  GPURasterEmbeddingDequantize: {
    limits: [
      {
        name: 'dimensions',
        maximum: 128,
        reason: 'One pixel thread keeps its whole vector in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-dequantize.ts'
      }
    ]
  },
  GPURasterEmbeddingQuantize: {
    limits: [
      {
        name: 'dimensions',
        maximum: 128,
        reason: 'One pixel invocation walks one bounded vector to determine validity.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-quantize.ts'
      }
    ]
  },
  GPURasterEmbeddingSimilarity: {
    limits: [
      {
        name: 'dimensions',
        maximum: 128,
        reason: 'One pixel thread keeps its whole vector in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity.ts'
      },
      {
        name: 'queryCount',
        maximum: 64,
        reason: 'Each pixel thread loops over every query.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity.ts'
      }
    ]
  },
  GPURasterEmbeddingNearestPixels: {
    limits: [
      {
        name: 'queryCount',
        maximum: 64,
        reason: 'The published exact search keeps a top-k per query invocation.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-nearest-pixels.ts'
      }
    ]
  },
  GPURasterEmbeddingChange: {
    limits: [
      {
        name: 'dimensions',
        maximum: 128,
        reason: 'One pixel thread keeps its whole vector in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-change.ts'
      }
    ]
  },
  GPURasterEmbeddingKMeans: {
    limits: [
      {
        name: 'maximumIterations',
        maximum: 64,
        reason: 'Every Lloyd iteration adds assignment and reduction passes to the graph.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts'
      }
    ]
  },
  GPURasterEmbeddingKNearestClassifier: {
    limits: [
      {
        name: 'trainingCount',
        maximum: 4096,
        reason: 'Every pixel thread scans the whole labelled set.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts'
      },
      {
        name: 'neighborCount',
        maximum: 16,
        reason: 'Nearest neighbours are kept in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts'
      },
      {
        name: 'classCount',
        maximum: 32,
        reason: 'Votes are counted in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts'
      }
    ]
  },
  GPURasterEmbeddingLinearProbe: {
    limits: [
      {
        name: 'classCount',
        maximum: 64,
        reason: 'Logits are held per pixel in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-linear-probe.ts'
      }
    ]
  },
  GPURasterEmbeddingCovariance: {
    limits: [
      {
        name: 'dimensions',
        maximum: 128,
        reason: 'One pixel thread keeps its whole vector in private memory.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-covariance.ts'
      },
      {
        name: 'rangeCount',
        maximum: 256,
        reason: 'Partial covariance holds rangeCount times dimensions squared floats.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-covariance.ts'
      }
    ]
  },
  GPURasterEmbeddingProjection: {
    limits: [
      {
        name: 'componentCount',
        maximum: 16,
        reason: 'Scores are computed per pixel in one invocation.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection.ts'
      }
    ]
  },
  GPURasterEmbeddingZoneMean: {
    limits: [
      {
        name: 'zoneCount',
        maximum: 65536,
        reason: 'The fixed-point accumulator of 128 bands fits one 128 MiB storage binding.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean.ts'
      }
    ]
  },
  GPURasterEmbeddingPyramidLevel: {
    limits: [
      {
        name: 'scale',
        maximum: 16,
        reason: 'Each overview pixel thread loops over its footprint and the vector.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level.ts'
      }
    ]
  },
  GPURasterStreamingAccumulator: {
    limits: [
      {
        name: 'chunkSliceCount',
        maximum: 256,
        reason: 'One invocation per cell loops over the chunk; the stream has no slice limit.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts'
      },
      {
        name: 'slotCount',
        maximum: 366,
        reason: 'Calendar slots per cell; each adds a loop step per chunk.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts'
      }
    ]
  },
  GPURasterRollingMean: {
    limits: [
      {
        name: 'windowLength',
        maximum: 128,
        reason: 'Each output sample rescans the window on the thread.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-rolling-mean.ts'
      }
    ]
  },
  GPURasterWhittakerSmoother: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason: 'One invocation per cell runs a banded Cholesky over private arrays of this size.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother.ts'
      },
      {
        name: 'reweightPassCount',
        maximum: 8,
        reason: 'Passes are unrolled into the per-cell loop.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother.ts'
      }
    ]
  },
  GPURasterHANTS: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason:
          'Per-thread QR working memory; total floats bounded at 2048 with sliceCount times coefficients.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-hants.ts'
      },
      {
        name: 'rejectionPassCount',
        maximum: 16,
        reason: 'Reject-and-refit passes are unrolled into the per-cell loop.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-hants.ts'
      }
    ]
  },
  GPURasterHarmonicRegression: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason:
          'Per-thread QR working memory; total floats bounded at 2048 with sliceCount times coefficients.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression.ts'
      }
    ]
  },
  GPURasterCOLDMonitor: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason:
          'Per-thread working memory; 8 coefficients allow 167 slices at 5 bands, 4 coefficients allow 256.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts'
      },
      {
        name: 'bandCount',
        maximum: 8,
        reason: 'Per-thread coefficient and residual arrays grow with the band count.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts'
      },
      {
        name: 'maximumSegmentCount',
        maximum: 32,
        reason: 'The per-cell output record is maximumSegmentCount long.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts'
      },
      {
        name: 'sliceCount * coefficientCount^2 * 2 * bandCount * maximumFitCount',
        maximum: 16777216,
        reason: 'Refits of one cell run inside one invocation.',
        source: 'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts'
      }
    ]
  },
  GPURasterBFASTMonitor: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason: 'Per-thread QR working memory; the history design is bounded at 2048 floats.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-bfast-monitor.ts'
      }
    ]
  },
  GPURasterSnapshotCovariance: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason: 'The dense temporal matrix grows with the square of the slice count.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-snapshot-covariance.ts'
      }
    ]
  },
  GPURasterEOFProjection: {
    limits: [
      {
        name: 'modeCount',
        maximum: 32,
        reason: 'Each mode adds a pass over the slices per cell.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-eof-projection.ts'
      }
    ]
  },
  GPURasterLaggedCorrelation: {
    limits: [
      {
        name: 'lags',
        maximum: 64,
        reason: 'Each lag adds a pass over the slices per cell.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-lagged-correlation.ts'
      }
    ]
  },
  GPURasterTemporalResample: {
    limits: [
      {
        name: 'gridCount',
        maximum: 4096,
        reason: 'A per-cell merge scans the grid in order.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-temporal-resample.ts'
      }
    ]
  },
  GPURasterMedoidComposite: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 64,
        reason:
          'One invocation compares every pair of valid observations and holds their validity in two 32-bit words.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-medoid-composite.ts'
      },
      {
        name: 'bandCount',
        maximum: 16,
        reason: 'The distance loop over bands runs inside the pair loop.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-medoid-composite.ts'
      }
    ]
  },
  GPURasterGeometricMedianComposite: {
    limits: [
      {
        name: 'iterationCount',
        maximum: 64,
        reason: 'Weiszfeld iterations are unrolled into the per-cell loop.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite.ts'
      },
      {
        name: 'sliceCount',
        maximum: 256,
        reason: 'Each iteration is one serial pass over the slices of the cell.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite.ts'
      }
    ]
  },
  GPURasterBestAvailablePixel: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason: 'One invocation scores every slice in turn.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-best-available-pixel.ts'
      }
    ]
  },
  GPURasterPhenologyMetrics: {
    limits: [
      {
        name: 'sliceCount',
        maximum: 256,
        reason: 'The curve times live in a private array and every season scans the curve.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics.ts'
      },
      {
        name: 'maximumSeasonCount',
        maximum: 2,
        reason: 'The season loop is unrolled and each season has private metric slots.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics.ts'
      }
    ]
  },
  GPUNetworkStronglyConnectedComponents: {
    tier: 'scale',
    limitation:
      'O(V) scratch. Deep graphs and chains of SCCs may exhaust the step budget; all labels then become 0xffffffff. CSR and scratch must fit device storage bindings. CPU reference: computeNetworkStructureOnCPU (Tarjan).',
    limits: [
      {
        name: 'maximumIterations',
        maximum: 4096,
        reason: 'Six command nodes per state-machine step fit the shared command node budget.',
        source:
          'modules/analysis/src/gpu-network/network-structure/gpu-network-strongly-connected-components.ts'
      },
      {
        name: 'nodeCount/edgeCount',
        maximum: 4294967294,
        reason:
          'uint32 indexing reserves 0xffffffff for invalid labels; device storage bindings impose a lower runtime limit.',
        source:
          'modules/analysis/src/gpu-network/network-structure/gpu-network-strongly-connected-components.ts'
      }
    ]
  },
  GPUNetworkStructure: {
    tier: 'reference',
    limits: [
      {
        name: 'nodeCount',
        maximum: 1024,
        reason:
          'Exact SCC scratch is O(V squared) and the resilience pass is a single serial invocation.',
        source: 'modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts'
      },
      {
        name: 'edgeCount',
        maximum: 65536,
        reason: 'Edge scratch is sized for this maximum.',
        source: 'modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts'
      }
    ]
  },
  GPUKShortestPaths: {
    tier: 'scale',
    limitation:
      'Yen (1971); lane-expanded CSR scratch scales with queries times path length. A shortest-search or path-length limit retains previously certified routes and sets iterationLimitReached; candidate exhaustion sets candidateOverflow. Equal-cost membership uses the reachability predecessor policy.',
    limits: [
      {
        name: 'routeCount',
        maximum: 32,
        reason: 'Bounds accepted-root comparisons and Yen rounds.',
        source: 'modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts'
      },
      {
        name: 'maximumPathLength',
        maximum: 1024,
        reason:
          'Bounds spur lanes, root comparisons and path reconstruction independently of CSR vertex count.',
        source: 'modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts'
      },
      {
        name: 'candidateCapacity',
        maximum: 4096,
        reason: 'Bounds the persistent candidate merge per query.',
        source: 'modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 1024,
        reason:
          'Matches the GPU shortest-path frontier and tie-phase limits; the shared 32768 command-node budget also applies.',
        source: 'modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts'
      }
    ]
  },
  GPUAlternativeRoutes: {
    tier: 'reference',
    limits: [
      {
        name: 'nodeCount',
        maximum: 256,
        reason: 'One invocation per query searches all simple paths with private state.',
        source: 'modules/analysis/src/gpu-network/alternative-routes/gpu-alternative-routes.ts'
      },
      {
        name: 'routeCount',
        maximum: 32,
        reason: 'Routes per query are held in per-invocation arrays.',
        source: 'modules/analysis/src/gpu-network/alternative-routes/gpu-alternative-routes.ts'
      }
    ]
  },
  GPUNetworkInhomogeneousK: {
    tier: 'reference',
    limits: [
      {
        name: 'eventCount',
        maximum: 1024,
        reason: 'Exhaustive all-pairs of events, so pairs grow with the square of the count.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'radiusCount',
        maximum: 256,
        reason: 'Per-radius accumulators are sized for this maximum.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'pairCount',
        maximum: 1048576,
        reason: 'Storage and work envelope of the explicit all-pairs path.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      }
    ]
  },
  GPUNetworkSnapping: {
    limits: [
      {
        name: 'positionsTimesEdgesWithoutCandidateCapacity',
        maximum: 134217728,
        reason: 'Without candidateCapacity every point tests every edge.',
        source: 'modules/analysis/src/gpu-network/network-accessibility/gpu-network-snapping.ts'
      }
    ]
  },
  GPUNetworkFlow: {
    tier: 'reference',
    limits: [
      {
        name: 'nodeCount',
        maximum: 128,
        reason: 'The augmenting-path solve runs in one invocation over a dense node set.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-flow.ts'
      },
      {
        name: 'edgeCount',
        maximum: 1024,
        reason: 'Arc arrays belong to the single solving invocation.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-flow.ts'
      }
    ]
  },
  GPUNetworkTravelTimePercentiles: {
    tier: 'reference',
    limits: [
      {
        name: 'departureCount',
        maximum: 60,
        reason:
          'The per-vertex sort is a private array and each departure is one routing dispatch.',
        source:
          'modules/analysis/src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles.ts'
      },
      {
        name: 'stateCount',
        maximum: 256,
        reason: 'Each earliest-arrival solve runs in one invocation over all states.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-time-routing.ts'
      },
      {
        name: 'transitionCount',
        maximum: 4096,
        reason: 'Each relaxation round of the single invocation scans every transition.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-time-routing.ts'
      }
    ]
  },
  GPUNetworkTimeRouting: {
    tier: 'reference',
    limits: [
      {
        name: 'stateCount',
        maximum: 256,
        reason: 'The earliest-arrival solve runs in one invocation over all states.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-time-routing.ts'
      },
      {
        name: 'transitionCount',
        maximum: 4096,
        reason: 'Transitions are scanned by the single solving invocation.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-time-routing.ts'
      }
    ]
  },
  GPUNetworkRationalAccess: {
    tier: 'reference',
    limits: [
      {
        name: 'demandCount',
        maximum: 64,
        reason:
          'Cooperative allocation cycles fit one 64-lane workgroup; initialization remains ordered.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-rational-access.ts'
      },
      {
        name: 'supplyCount',
        maximum: 32,
        reason:
          'Cooperative allocation cycles fit one 64-lane workgroup; initialization remains ordered.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-rational-access.ts'
      }
    ]
  },
  GPUNetworkLouvain: {
    tier: 'reference',
    limitation:
      'Reference tier applies to legacy dense weights (32 nodes). Supply topology for sparse parallel multilevel moving; up to 65,535 nodes and 1,048,576 CSR slots, subject to the command-node budget.',
    limits: [
      {
        name: 'nodeCount',
        maximum: 32,
        reason: 'The vertex-moving passes run in one invocation over a dense weight matrix.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-louvain.ts'
      },
      {
        name: 'nodeCount (CSR)',
        maximum: 65535,
        reason: 'Two node IDs share a uint32 pair key with a reserved padding sentinel.',
        source: 'modules/analysis/src/gpu-network/network-optimization/network-louvain-sparse.ts'
      },
      {
        name: 'CSR edge capacity',
        maximum: 1048576,
        reason: 'Bounds two weak-neighbor records per slot and sort/scan scratch.',
        source: 'modules/analysis/src/gpu-network/network-optimization/network-louvain-sparse.ts'
      },
      {
        name: 'estimatedCommandNodeCount (CSR)',
        maximum: 32768,
        reason: 'Construction estimates the unrolled local-moving and aggregation schedule.',
        source: 'modules/analysis/src/gpu-network/network-optimization/network-louvain-sparse.ts'
      }
    ]
  },
  GPUNetworkFacilityOptimization: {
    tier: 'reference',
    limits: [
      {
        name: 'facilityCount',
        maximum: 16,
        reason: 'Exhaustive subset enumeration, 2 to the facility count.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-facility-optimization.ts'
      },
      {
        name: 'demandCount',
        maximum: 64,
        reason: 'Per-candidate demand arrays are sized for this maximum.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-facility-optimization.ts'
      },
      {
        name: 'candidateCount',
        maximum: 1048576,
        reason:
          'Candidates are evaluated exhaustively: 2 to the facility count, or facilities to the power of demand rows for capacitated p-median.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-facility-optimization.ts'
      }
    ]
  },
  GPUNetworkPointRouting: {
    tier: 'reference',
    limits: [
      {
        name: 'edgeCount',
        maximum: 128,
        reason: 'Edge costs are held in per-query private arrays.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-point-routing.ts'
      },
      {
        name: 'turnCount',
        maximum: 4096,
        reason: 'Turns are scanned by each query invocation.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-point-routing.ts'
      }
    ]
  },
  GPUMarkedPointStatistics: {
    tier: 'reference',
    limits: [
      {
        name: 'rowCount',
        maximum: 16384,
        reason:
          'Only with searchStructure brute-force: dense reference traversal of all pairs, O(rows squared times radii).',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics.ts'
      },
      {
        name: 'rowCount x radiusCount x 3',
        maximum: 16777216,
        reason:
          'Grid search (the default): three float32 partials per row and radius; device binding limits also apply.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics.ts'
      },
      {
        name: 'radiusCount',
        maximum: 128,
        reason: 'Per-radius accumulators are sized for this maximum.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics.ts'
      }
    ]
  },
  GPUCrossVariogram: {
    tier: 'reference',
    limits: [
      {
        name: 'rowCount',
        maximum: 16777215,
        reason:
          'Row IDs and counts must remain exact in float32; device binding limits also apply.',
        source: 'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-cross-variogram.ts'
      },
      {
        name: 'rowCount x binCount',
        maximum: 16777216,
        reason:
          'Default and explicit lattices share this partial-table budget; pairs come from a cell index.',
        source: 'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-cross-variogram.ts'
      },
      {
        name: 'binCount',
        maximum: 128,
        reason: 'Lag bins times time bins are accumulated in per-invocation arrays.',
        source: 'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-cross-variogram.ts'
      }
    ]
  },
  GPUInequality: {
    tier: 'reference',
    limitation:
      'The per-zone walk is a serial loop over the zone rows on one invocation, a bottleneck at about 2^20 rows in one zone, and the float32 running sums stop matching float64 references past about 2^22 rows.',
    limits: [
      {
        name: 'lorenzKnotCount',
        maximum: 1025,
        reason: 'Lorenz curve knots are held in per-zone arrays.',
        source: 'modules/analysis/src/dataframe-analysis/composite-indicators/gpu-inequality.ts'
      }
    ]
  },
  GPUSpatialBalancedSample: {
    tier: 'reference',
    limits: [
      {
        name: 'rows',
        maximum: 4096,
        reason: 'The per-stratum selection pass is serial.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample.ts'
      },
      {
        name: 'strata',
        maximum: 256,
        reason: 'The per-stratum selection pass is serial.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample.ts'
      }
    ]
  },
  GPUInhomogeneousPointPattern: {
    tier: 'reference',
    limits: [
      {
        name: 'events',
        maximum: 4096,
        reason: 'The intensity and pair kernels are all-pairs, O(n squared).',
        source:
          'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern.ts'
      },
      {
        name: 'radii',
        maximum: 256,
        reason: 'Radii are evaluated in a single resident curve.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern.ts'
      },
      {
        name: 'overlapEdgePairs',
        maximum: 1048576,
        reason: 'Bounds the per-pair triangle-intersection loop of the overlap term.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern.ts'
      }
    ]
  },
  GPUSpatialCrossValidation: {
    tier: 'reference',
    limits: [
      {
        name: 'rows',
        maximum: 65536,
        reason:
          'Each buffer-mask invocation scans all rows, so work grows as folds times rows squared.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts'
      },
      {
        name: 'foldCount',
        maximum: 64,
        reason: 'Masks are fold-major, rows times folds entries each.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts'
      },
      {
        name: 'designRows',
        maximum: 256,
        reason: 'Fold designs run as one serial invocation each (64 rows for NNDM).',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/spatial-validation-design.ts'
      }
    ]
  },
  GPUS2CapCovering: {
    tier: 'reference',
    limitation:
      "The default 'uniform' enumeration tests every cell at the level in one invocation per cap, so the level is at most 8. The option refinement: 'level-synchronous' prunes by level in parallel and reaches level 15 with the same cells.",
    limits: [
      {
        name: 'uniformLevel',
        maximum: 8,
        reason:
          "Each cap's invocation tests every cell at the level (default uniform enumeration).",
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering.ts'
      }
    ]
  },
  GPUS2PolygonCovering: {
    tier: 'reference',
    limitation:
      "The default 'serial' traversal runs in one invocation (an O(V squared) ring check, then a depth-first walk of up to 1,048,576 cells). The option refinement: 'level-synchronous' distributes both across invocations.",
    limits: [
      {
        name: 'maximumVisitedCells',
        maximum: 1048576,
        reason: 'Bounds the visited-cell walk of the traversal.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering.ts'
      }
    ]
  },
  GPURasterSpectralFilter: {
    limits: [
      {
        name: 'grid dimension',
        maximum: 2048,
        reason:
          'Published GPUFFT2D radix-two limit; each complex field must fit a device storage binding.',
        source: 'modules/analysis/src/raster-analysis/spectral-filter/gpu-raster-spectral-filter.ts'
      }
    ],
    limitation:
      'Complete periodic resident grid; any missing input invalidates the entire filtered field. Frequencies are cycles per pixel.'
  },
  GPURasterRegistration: {
    limits: [
      {
        name: 'periodicDimension',
        maximum: 2048,
        reason: 'Three complex scratch fields of side squared must fit one 128 MiB binding.',
        source: 'modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-registration.ts'
      },
      {
        name: 'maskedDimension',
        maximum: 512,
        reason:
          'Six padded correlation fields (about twice the side, rounded to a power of two) plus FFT scratch must fit device storage; the masked statistics and refinement sums are serial per cell.',
        source: 'modules/analysis/src/raster-analysis/remote-sensing/raster-masked-registration.ts'
      }
    ]
  },
  GPURasterRegionGraph: {
    tier: 'reference',
    limits: [
      {
        name: 'regionCount',
        maximum: 256,
        reason: 'Region adjacency and merging run in one invocation over dense tables.',
        source: 'modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 256,
        reason: 'Merge rounds are unrolled into the command graph.',
        source: 'modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph.ts'
      }
    ]
  },
  GPURasterFractionalCoverage: {
    limits: [
      {
        name: 'zoneCountTimesCellCount',
        maximum: 4194304,
        reason: 'Dense zone by cell candidate grid; bounds the coverage and compaction scratch.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts'
      }
    ]
  },
  GPURasterFractionalZonalStatistics: {
    limits: [
      {
        name: 'zoneCountTimesCellCount',
        maximum: 4194304,
        reason: 'Shares the coverage construction of GPURasterFractionalCoverage.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts'
      }
    ]
  },
  GPUTerrainBreach: {
    tier: 'reference',
    limits: [
      {
        name: 'scratchWords',
        maximum: 4294967295,
        reason:
          'Search scratch is pits times (steps + 1) times the search window, two words per cell, and must be addressable with 32 bits; it is not checked against the storage binding.',
        source: 'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-breach.ts'
      }
    ]
  },
  GPUTerrainDepressionHierarchy: {
    tier: 'reference',
    limitation:
      'The merge-tree build is serial: one GPU invocation walks the sorted cells with union-find, so time grows linearly with the cell count and very large grids can approach GPU watchdog limits.'
  },
  GPUTerrainDEMCoregistration: {
    tier: 'reference',
    limits: [
      {
        name: 'grid cell count with correlationLength > 0',
        maximum: 92672,
        reason:
          'Two covariance partial planes each hold cells x ceil(cells / 256) float32 values. The next cell exceeds the core 128 MiB storage-binding limit; this cap also bounds uint32 partial indices. Independent errors are not subject to this cap.',
        source:
          'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts'
      }
    ],
    limitation:
      'Nonzero correlation evaluates the full quadratic covariance model in bounded 256-cell tiles with fixed-order segmented reductions. Correlated grids are capped at 92,672 cells at construction; independent errors keep linear work and scratch.'
  },
  GPUPointCloudNeighborhoods: {
    limits: [
      {
        name: 'capacity',
        maximum: 64,
        reason: 'Neighbor lists are held in per-query arrays.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      },
      {
        name: 'queriesTimesPositionsWithoutGrid',
        maximum: 134217728,
        reason: 'Without a grid every query tests every point.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudNormalScale: {
    limitation:
      'Minimum population plane roughness over supplied radii, excluding the query itself. Truncated scales are rejected; grid coverage must include the intended support.',
    limits: [
      {
        name: 'radii.length',
        maximum: 32,
        reason: 'Three graph kernels per scale.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      },
      {
        name: 'capacity',
        maximum: 64,
        reason: 'Per-query neighbor arrays.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      },
      {
        name: 'queriesTimesPositionsWithoutGrid',
        maximum: 134217728,
        reason: 'Exhaustive search without a grid.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudChange: {
    limits: [
      {
        name: 'samplingCovariancePoints',
        maximum: 32768,
        reason: 'One thread visits every pair of in-cylinder points, quadratic per core.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      },
      {
        name: 'coresTimesPointsWithoutGrid',
        maximum: 134217728,
        reason: 'Without a grid every core scans both clouds.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudICP: {
    limits: [
      {
        name: 'iterations',
        maximum: 100,
        reason: 'Iterations are unrolled into the command graph.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      },
      {
        name: 'sourceTimesTargetWithoutGrid',
        maximum: 134217728,
        reason: 'Without a target grid every source point tests every target point.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudSMRF: {
    limits: [
      {
        name: 'windowCells',
        maximum: 64,
        reason: 'Morphology window radius in cells; cost grows with the window.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      },
      {
        name: 'cutCells',
        maximum: 32,
        reason: 'Cut radius in cells.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudGround: {
    limits: [
      {
        name: 'radiusCells',
        maximum: 32,
        reason: 'Largest morphology radius in cells.',
        source: 'modules/analysis/src/pointcloud-analysis/pointcloud-limits.ts'
      }
    ]
  },
  GPUFuzzyContiguity: {
    tier: 'reference',
    limits: [
      {
        name: 'features',
        maximum: 65536,
        reason:
          'Each emitted pair is sorted by one uint32 key, 16 bits of row and 16 bits of neighbor; per-row neighbor counts and the required total stay within uint32.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts'
      },
      {
        name: 'vertices',
        maximum: 65536,
        reason: 'Vertices are scanned for every candidate pair the bounds index returns.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts'
      }
    ]
  },
  GPUInstrumentalVariablesCovariance: {
    limits: [
      {
        name: 'coefficientCount',
        maximum: 15,
        reason:
          'The covariance PSD certificate factors at most 16 coefficients including the implicit intercept.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance.ts'
      },
      {
        name: 'instrumentCount',
        maximum: 16,
        reason: 'Centered projection uses a bounded small dense factorization.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance.ts'
      },
      {
        name: 'clusterCount',
        maximum: 16777216,
        reason:
          'Dense cluster-key capacity cannot exceed rows; occupied groups alone enter CR1. Device score binding limits also apply.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance.ts'
      },
      {
        name: 'rows',
        maximum: 16777216,
        reason:
          'Rows are tiled across workgroups; counts are float32 and exact only to 2 to the 24th.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance.ts'
      }
    ]
  },
  GPUSpatialPanelLagMaximumLikelihood: {
    tier: 'reference',
    limits: [
      {
        name: 'entities',
        maximum: 32,
        reason: 'A dense entity by entity determinant is formed per evaluation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts'
      },
      {
        name: 'periodCount',
        maximum: 32,
        reason: 'Panel periods are unrolled into the kernel.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts'
      }
    ]
  },
  GPUProbitRegression: {
    limits: [
      {
        name: 'rows',
        maximum: 16777216,
        reason:
          'Rows are tiled across workgroups; counts are float32 and exact only to 2 to the 24th.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression.ts'
      }
    ]
  },
  GPUSpatialRegimeRegression: {
    limits: [
      {
        name: 'rows',
        maximum: 16777216,
        reason:
          'Rows are tiled across workgroups; counts are float32 and exact only to 2 to the 24th.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression.ts'
      },
      {
        name: 'regimeCount',
        maximum: 4096,
        reason: 'Regime moments are packed per regime in one bounded layout.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression.ts'
      }
    ]
  },
  GPURestrictedRegimeRegression: {
    limits: [
      {
        name: 'rows',
        maximum: 16777216,
        reason:
          'Rows are tiled across workgroups; counts are float32 and exact only to 2 to the 24th.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts'
      },
      {
        name: 'regimeCount',
        maximum: 8,
        reason: 'Regime blocks are dense and held per invocation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts'
      }
    ]
  },
  GPUGeneralizedGeographicallyWeightedRegression: {
    tier: 'reference',
    limitation:
      'Unpenalized GLM; separation is a sufficient float32 recession-direction certificate, not an exhaustive LP test. AICc needs a positive small-sample denominator.',
    limits: [
      {
        name: 'response rows (dense)',
        maximum: 65536,
        reason: 'Dense local IRLS scans a bounded training neighborhood per invocation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'response rows (indexed)',
        maximum: 1048576,
        reason:
          'Indexed local fits bound row IDs and scratch; exact unbounded Gaussian still scans all rows.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'bandwidthSelection AICc rows',
        maximum: 4096,
        reason: 'Repeated dense IRLS and AICc selection are bounded reference work.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'maximumNeighborCount',
        maximum: 128,
        reason: 'Adaptive distance order statistics occupy a per-invocation list.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'predictorCount',
        maximum: 24,
        reason: 'Per-invocation dense information matrices and solves.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 128,
        reason: 'Bounded local Newton/IRLS steps.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'bandwidthSelection.maximumIterations',
        maximum: 64,
        reason: 'Golden-section rounds expand into gated command nodes.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      }
    ]
  },
  GPUMultiscaleGeographicallyWeightedRegression: {
    limits: [
      {
        name: 'denseRows (indexGridSize false, untruncated Gaussian or great-circle)',
        maximum: 65536,
        reason:
          'Dense local fits scan all included rows; planar finite-support kernels index automatically from 1024 rows.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts'
      },
      {
        name: 'indexedRows',
        maximum: 1048576,
        reason:
          'Planar finite-support fits visit indexed cells; selection scratch must also fit one storage binding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts'
      },
      {
        name: 'maximumSearchIterations',
        maximum: 64,
        reason:
          'Per-component golden-section search unrolls command nodes; the shared 32768-node budget also applies.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/mgwr-bandwidth-selection.ts'
      },
      {
        name: 'inferenceRows',
        maximum: 128,
        reason:
          'Only with full inference: the dense hat-matrix backfit costs rows cubed times predictors times iterations.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts'
      }
    ]
  },
  GPUOPTICS: {
    tier: 'reference',
    limits: [
      {
        name: "rows (neighborIndex 'brute-force')",
        maximum: 4096,
        reason: 'Every visit scans every row in the sequential exhaustive reference.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts'
      },
      {
        name: "rows (neighborIndex 'grid', the default above 4096)",
        maximum: 65536,
        reason:
          'Indexed relaxation; orderingMethod parallel distributes exact frontier selection and relaxation across workgroups with at most nine nodes per dependent visit. Xi accepts this limit with 64-rank sequential extraction batches.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts'
      }
    ]
  },
  GPUGaussianRandomField: {
    tier: 'scale',
    limitation:
      'Planar isotropic fields. FFT clips negative eigenvalues and reports variance mass. Finite modes and indexed local residual conditioning are approximations. Spherical covariance requires FFT. Conditioning sites are rows of the output domain.',
    limits: [
      {
        name: 'FFT field side',
        maximum: 1024,
        reason: 'Doubled power-of-two embedding fits published GPUFFT2D at 2048 per side.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field.ts'
      },
      {
        name: 'modeCount',
        maximum: 4096,
        reason: 'Bounded per-point spectral sum; total mode-point work is capped at 2^30.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field.ts'
      },
      {
        name: 'rowCount',
        maximum: 2 ** 24,
        reason:
          'Packed float32x2 coordinates fit 128 MiB; scratch and candidate bindings are also device-checked.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field.ts'
      }
    ]
  },
  GPULocalKriging: {
    limits: [
      {
        name: 'neighborCount',
        maximum: 32,
        reason: 'The neighbor list and the dense local system are held per query invocation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts'
      },
      {
        name: 'positions.length (default exhaustive scan)',
        maximum: 4096,
        reason:
          "sampleIndex 'auto' scans every sample up to this count and takes candidates from a GPUNeighborSearch grid above it.",
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts'
      },
      {
        name: "samples * queries (any exhaustive scan, explicit or chosen by sampleIndex 'auto')",
        maximum: 2 ** 32,
        reason:
          "Each query invocation scans every sample; the exhaustive-scan work ceiling. 'auto' takes the grid above it.",
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts'
      }
    ]
  },
  GPUConditionalGaussianSimulation: {
    tier: 'reference',
    limits: [
      {
        name: 'observationsPlusQueries',
        maximum: 4096,
        reason: 'Dense Cholesky factor of the joint covariance, 64 MiB at the maximum.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts'
      }
    ]
  },
  GPUBrownianBridgeUtilization: {
    tier: 'reference',
    limits: [
      {
        name: 'fix count',
        maximum: 16384,
        reason:
          'Every grid cell loops over all fixes (work is cell count times fix count); no segment index is built.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization.ts'
      },
      {
        name: 'subStepCount',
        maximum: 256,
        reason: 'Time steps per segment are a bounded per-thread loop.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization.ts'
      }
    ]
  },
  GPUUtilizationVolumeIsopleths: {
    limits: [
      {
        name: 'cellCount',
        maximum: 16777216,
        reason: 'Three prefix levels of 256 rows each, the last one a single invocation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths.ts'
      }
    ]
  },
  GPUTrajectoryRecursions: {
    tier: 'reference',
    limits: [
      {
        name: 'maximumTrackSegments',
        maximum: 65536,
        reason:
          'Each fix and radius loops over the segments of its track (quadratic in track length).',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions.ts'
      },
      {
        name: 'radius count',
        maximum: 32,
        reason: 'One output column per radius in the same pass.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions.ts'
      }
    ]
  },
  GPUScalarFieldRidges: {
    limits: [
      {
        name: 'grid axis length',
        maximum: 65535,
        reason:
          'Bounds row-major uint32 addressing; the full grid must also fit a storage binding.',
        source:
          'modules/analysis/src/raster-analysis/lagrangian-diagnostics/gpu-scalar-field-ridges.ts'
      }
    ]
  },
  GPULagrangianFlowMap: {
    limits: [
      {
        name: 'stepCount',
        maximum: 8192,
        reason: 'Each particle takes all of its steps inside one invocation (a bounded loop).',
        source:
          'modules/analysis/src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map.ts'
      }
    ]
  },
  GPURadiationFlowModel: {
    tier: 'reference',
    limits: [
      {
        name: 'zoneCount',
        maximum: 1024,
        reason:
          'Each origin-destination pair sums the populations of all nearer zones (cubic in zone count); the scalable form needs a float segmented scan.',
        source: 'modules/analysis/src/gpu-network/flow-aggregation/gpu-radiation-flow-model.ts'
      }
    ]
  },
  GPUKernelFlowSmoothing: {
    tier: 'reference',
    limits: [
      {
        name: 'zoneCount',
        maximum: 1024,
        reason:
          'Neighbourhood radii and both smoothing passes sum over all zones for each flow (cubic in zone count).',
        source: 'modules/analysis/src/gpu-network/flow-aggregation/gpu-kernel-flow-smoothing.ts'
      }
    ]
  },
  GPURankMarkov: {
    tier: 'reference',
    limits: [
      {
        name: 'rowCount',
        maximum: 64,
        reason: 'The transition matrix is dense and held per invocation.',
        source: 'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts'
      }
    ]
  },
  GPUJacquezTest: {
    limits: [
      {
        name: 'k',
        maximum: 256,
        reason:
          'The k temporal nearest neighbors of an event are held in per-invocation registers.',
        source: 'modules/analysis/src/gpu-spatial-analysis/space-time-tests/gpu-jacquez-test.ts'
      }
    ]
  },
  GPUSpaceTimeKFunction: {
    limits: [
      {
        name: 'positions length',
        maximum: 8388608,
        reason: 'The row table holds four float32 per row in one 128 MiB storage binding.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/space-time-k-function-nodes.ts'
      },
      {
        name: 'distanceClassCount',
        maximum: 64,
        reason:
          'Distance classes share the 2048 two-word accumulators of the workgroup pair histogram with the time classes.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-space-time-k-function.ts'
      }
    ]
  },
  GPUSpaceTimeDiggleTest: {
    limits: [
      {
        name: 'permutationCount',
        maximum: 999,
        reason:
          'The statistics kernels loop over the permutations inside one invocation per class.',
        source:
          'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test.ts'
      }
    ]
  },
  GPUSpaceTimeCrossCorrelation: {
    limits: [
      {
        name: 'lags',
        maximum: 64,
        reason: 'Every lag keeps its own correlation sums in the invocation of a cell.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation.ts'
      }
    ]
  },
  GPUMobilityIndicators: {
    limits: [
      {
        name: 'entropySequenceRows',
        maximum: 16384,
        reason:
          'The real-entropy estimator is cubic in the staypoints of a track in the worst case, so the location sequence is capped.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators.ts'
      }
    ]
  },
  GPUTrajectoryLocations: {
    limits: [
      {
        name: 'maximumIterations',
        maximum: 256,
        reason: 'Component labels are hooked in a bounded number of gated rounds.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations.ts'
      }
    ]
  },
  // Internal contributor (not a public operator): recorded here for contract rule 12, so its
  // limits stay beside the others until it is exported.
  GPUSnapRoundedNoding: {
    limitation:
      'Planar only. lines.positions rows, and 4 x (rows + intersectionCapacity), are at most the device segment-intersection vertex limit (getGPUSegmentIntersectionMaximumVertexCount); leafCapacity must be a power of two. Incomplete output sets candidateOverflow or overflow.'
  },
  GPUNestedInterpolationValidation: {
    tier: 'reference',
    limits: [
      {
        name: 'rows',
        maximum: 256,
        reason: 'The outer-mask overlap check and each fold fit run in single invocations.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts'
      },
      {
        name: 'foldCount',
        maximum: 16,
        reason: 'Outer folds are unrolled into separate command nodes.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts'
      },
      {
        name: 'innerFoldCount',
        maximum: 8,
        reason: 'Inner folds are unrolled per candidate inside each outer fold.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts'
      },
      {
        name: 'candidates',
        maximum: 16,
        reason: 'The hyperparameter ladder is searched exhaustively.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts'
      },
      {
        name: 'neighborCount',
        maximum: 32,
        reason: 'Each local system is a bounded dense solve.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts'
      }
    ]
  },
  addResistanceConnectivityRecipe: {
    tier: 'reference',
    limits: [
      {
        name: 'vertexCount',
        maximum: 256,
        reason: 'One solver thread keeps five private arrays of one value per vertex.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/resistance-connectivity.ts'
      },
      {
        name: 'edges.length',
        maximum: 65536,
        reason: 'The single solver thread sweeps every edge in every conjugate gradient iteration.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/resistance-connectivity.ts'
      }
    ]
  },
  addSparseResistanceConnectivityRecipe: {
    limits: [
      {
        name: 'vertexCount',
        maximum: 1048576,
        reason: 'The vertex-wide reduction trees and label arrays stay within one binding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-connectivity.ts'
      },
      {
        name: 'edges.length',
        maximum: 2097152,
        reason: 'Twice as many half-edge entries stay within one 128 MiB binding of uint32 pairs.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-connectivity.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 4096,
        reason: 'Conjugate gradient iterations are unrolled, five command nodes each.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-connectivity.ts'
      }
    ]
  },
  addRasterResistanceConnectivityRecipe: {
    limits: [
      {
        name: 'grid cells (width * height)',
        maximum: 1048576,
        reason: 'Every cell is a vertex of the sparse resistance solve.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts'
      },
      {
        name: 'edge slots (grid cells * 2 for neighborhood 4, * 4 for neighborhood 8)',
        maximum: 2097152,
        reason: 'Every edge slot is an edge row of the sparse resistance solve.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 4096,
        reason: 'Conjugate gradient iterations are unrolled, five command nodes each.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts'
      }
    ]
  },
  GPUParameterBuffer: {
    role: 'helper'
  },
  GPURasterContrastEqualization: {
    tier: 'scale',
    limitation:
      'Quantized distributions; explicit input/reference domains. CLAHE follows scikit-image 0.24 clipping and interpolation, without its global output rescale. Reference: https://scikit-image.org/docs/0.24.x/api/skimage.exposure.html',
    limits: [
      {
        name: 'grid cell count (including reflected tile padding)',
        maximum: 16777216,
        reason: '64 MiB scalar binding and exact float32 counts.',
        source:
          'modules/analysis/src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts'
      },
      {
        name: 'histogramLength',
        maximum: 16777216,
        reason: '64 MiB histogram, scan and mapping bindings.',
        source:
          'modules/analysis/src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts'
      },
      {
        name: 'binCount',
        maximum: 256,
        reason: 'Bounded serial redistribution per tile.',
        source:
          'modules/analysis/src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts'
      },
      {
        name: 'tileWidth/tileHeight',
        maximum: 256,
        reason: 'At most 65,536 virtual pixels per tile.',
        source:
          'modules/analysis/src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts'
      }
    ]
  }
};
