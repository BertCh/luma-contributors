// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisLimit, GPUAnalysisTier} from '../spatial-analysis-ontology';

/** Audited tier, limits and role; omitted fields carry no inference about unaudited operators. */
export type OperatorAnnotation = {
  tier?: GPUAnalysisTier;
  limits?: readonly GPUAnalysisLimit[];
  limitation?: string;
  role?: 'operator' | 'helper';
};

/**
 * Limits and tiers confirmed against the source on 2026-11-15 (each `source` is the file that
 * enforces the maximum). A `reference` operator is a single-invocation, exhaustive or small-dense
 * design with a hard maximum size. An omitted tier is unaudited for newer entry points. Several of these contributors
 * are being reworked, so a limit that disappears from its source must be removed here as well.
 */
export const OPERATOR_ANNOTATIONS: Readonly<Record<string, OperatorAnnotation>> = {
  ...Object.fromEntries(
    [
      ['GPUSkaterRegions', 'gpu-skater-regions.ts'],
      ['GPUWardRegions', 'gpu-ward-regions.ts'],
      ['GPUAZPRegions', 'gpu-azp-regions.ts'],
      ['GPUMaxPRegions', 'gpu-max-p-regions.ts']
    ].map(([exportName, filename]) => [
      exportName,
      {
        limits: [
          {
            name: 'estimatedCommandNodeCount',
            maximum: 32768,
            reason:
              'GPU_MAXIMUM_COMMAND_NODE_COUNT: construction admission is checked by validateCommandNodeBudget for the selected rows, capacities and iteration/region budgets.',
            source: `modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/${filename}`
          }
        ],
        limitation:
          exportName === 'GPUSkaterRegions'
            ? 'The construction estimate must be at most 32,768 command nodes. Compiles min(maximumRegionCount - 1, rows - 1) attainable cuts; optional cut logs retain maximumRegionCount - 1 entries with sentinel/zero padding.'
            : 'The construction estimate must be at most 32,768 command nodes; admission depends on rows and the requested iteration/region budgets, not a universal row ceiling.'
      }
    ])
  ),
  GPURasterTileReprojection: {
    limits: [
      {
        name: 'sources.length',
        maximum: 16382,
        reason:
          'Derived from 3 + 2 * sources.length <= GPU_MAXIMUM_COMMAND_NODE_COUNT (32768); the validated source plan is snapshotted at construction.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-tile-reprojection.ts'
      },
      {
        name: 'estimatedCommandNodeCount',
        maximum: 32768,
        reason:
          'GPU_MAXIMUM_COMMAND_NODE_COUNT: validateCommandNodeBudget enforces construction admission.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-tile-reprojection.ts'
      }
    ],
    limitation:
      'estimatedCommandNodeCount = 3 + 2 * sources.length must fit the 32,768-node construction budget; source tiles and bounded mosaic scratch also remain subject to device storage bindings.'
  },
  GPUTerrainDerivatives: {
    limitation:
      'Orthogonal pixel axes are required: the constructor and parameter packer reject shear (normalized axis dot product tolerance 1e-6). Rotated orthogonal grids are supported only with uniform cell sizes; latitude-dependent ground modes require axis-aligned grids.'
  },
  GPURasterZonalStatistics: {
    limitation:
      'Geographic cell weighting requires an axis-aligned grid. Parameter layout version 2 explicitly carries signed row placement; spare parameter capacity does not select that layout.'
  },
  GPUAnalysisContiguityStatistics: {
    tier: 'scale',
    limits: [
      {
        name: 'featureCount',
        maximum: 16777216,
        reason: 'Converted cardinalities and default column quantile ranks use float32 integers.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.ts'
      }
    ],
    limitation:
      'Exact for supplied float32 equality/snap topology; positive snapping requires coordinateExtent. Matching rook edges need identical endpoint segmentation. Median is rounded to float32. Both candidate and slot capacities must fit; validity is zero otherwise.'
  },
  GPUAnalysisSpacing: {
    tier: 'scale',
    limitation:
      'Full-population indexed f32 spacing; geographic distances use the mean Earth sphere in metres. Headless numerical and timing qualification pending.',
    limits: [
      {
        name: 'positions.length',
        maximum: 1000000,
        reason: 'Bounds resident scratch and keeps count arithmetic exact in float32.',
        source: 'modules/analysis/src/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.ts'
      },
      {
        name: 'neighborRank',
        maximum: 64,
        reason: 'Bounded per-row k-neighbour lists, shared with the CPU spacing reference.',
        source: 'modules/analysis/src/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.ts'
      },
      {
        name: 'neighborCapacity',
        maximum: 33554432,
        reason: '128 MiB guaranteed WebGPU storage binding size.',
        source: 'modules/analysis/src/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.ts'
      }
    ]
  },
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
        maximum: 4194304,
        reason:
          'Default BVH core distances; compact BVH and tree scratch are device-binding bound, up to 4,194,304 rows.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts'
      },
      {
        name: 'minimumSamples',
        maximum: 32,
        reason: 'Bounds the indexed per-query top-k list.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts'
      },
      {
        name: 'positions.length (coreDistanceSearch neighbor-search)',
        maximum: 1048576,
        reason: 'Explicit packed CSR reference; planar points without times only.',
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
      'WGS84 inverse within 0.02 radians of an antipode uses a sphere fallback with converged=0 and approximation=1. Spherical midpoints within 1e-6 radians of an antipode are NaN. See docs/api/gpu-spatial-analysis/geodesic-precision.md for measured arithmetic envelopes and pending GPU certification. Aligned GraphVectorView origin, target and output rows support chunked pairs; iteration and precision limits remain per pair.'
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
      'Float32 coordinates and bounded Vincenty iteration; unconverged results are published with iterationLimitReached=1. Nonzero-distance arrivals with abs(cos(latitude)) <= 0.03 report approximation=1 because azimuth is ill-conditioned. WGS84 and sphere envelopes are documented in docs/api/gpu-spatial-analysis/geodesic-precision.md; arbitrary custom ellipsoids are outside those measurements. Aligned GraphVectorView origin, bearing, distance and output rows support chunked input; iteration and precision limits remain per row.'
  },
  ...Object.fromEntries(
    [
      'GPUGeographicallyTemporallyWeightedRegression',
      'GPUMultiscaleGeographicallyTemporallyWeightedRegression'
    ].map(exportName => [
      exportName,
      {
        tier:
          exportName === 'GPUGeographicallyTemporallyWeightedRegression'
            ? ('scale' as const)
            : ('reference' as const),
        limits: [
          ...(exportName === 'GPUGeographicallyTemporallyWeightedRegression'
            ? [
                {
                  name: 'rows (indexed finite-support)',
                  maximum: 16777216,
                  reason:
                    'Default grid for bisquare or explicit Gaussian cutoff without bandwidthSearch; device-binding bound.',
                  source:
                    'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression.ts'
                }
              ]
            : []),
          {
            name: 'rows (reference)',
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
          exportName === 'GPUGeographicallyTemporallyWeightedRegression'
            ? 'Planar native metric; default grid for finite support without bandwidthSearch, device-binding bound up to 16,777,216 rows. Explicit brute-force, untruncated Gaussian and bandwidth search retain the 32,768-row reference. Gaussian cutoff truncates support; output.status reports profile-search convergence and requires bandwidthSearch. Profile search assumes unimodality and adaptive ties can be nonsmooth.'
            : 'Planar native metric; multiscale initialization explicitly retains the 32,768-row reference. searchStructure and gaussianCutoff are unsupported. Golden-section profile search assumes unimodality, tests endpoints and reports interval convergence; adaptive neighbour ties can make the objective nonsmooth.'
      }
    ])
  ),
  GPUForestInference: {
    tier: 'scale',
    limitation:
      'Numerical splits and constant leaves only; caller-supplied models, float32 features and scores. Aligned GraphVectorView feature and output records support chunked batches; feature-major chunks contain all columns for their batch. No DART, categorical splits, multi-target trees or fitting.',
    limits: [
      ...[
        ['treeCount', 1024, 'Bounded traversal per row.'],
        ['maximumDepth', 64, 'Bounded sparse tree walk.'],
        ['nodeCount', 1048576, 'Sparse node storage is at most 24 MiB.'],
        ['featureCount', 256, 'Bounded feature index.'],
        ['outputCount', 32, 'Per-invocation score array.'],
        [
          'rowCount',
          4294967295,
          'Total rows use uint32; aligned GraphVectorView batches and reusable scratch must fit device bindings.'
        ],
        ['rasterStack.bands.length', 16, 'One preparation dispatch per band.'],
        [
          'worst-case work',
          5000000000,
          'Per-batch traversal and score accumulation work; split aligned row records into batches.'
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
    limitation:
      'Auto scans up to 4096 samples and uses the metric grid above that count. Aligned GraphVectorView queries, queryTimes and outputs support chunked query records; shared samples remain packed. Local solver and candidate-list caps remain.',
    tier: 'scale',
    limits: [
      {
        name: 'positions.length',
        maximum: 1073741808,
        reason:
          'Indexed samples are device-binding bound, up to 1,073,741,808; auto selects the exact metric grid above 4096 samples.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts'
      },
      {
        name: 'positions.length (sampleIndex brute-force)',
        maximum: 1048576,
        reason: 'Explicit exhaustive reference scans every sample per query.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts'
      }
    ]
  },
  GPURasterFalseDiscoveryRateMask: {
    limitation:
      'Exact integer ranks support at most 2,147,483,648 cells, subject to sort and device binding limits.',
    limits: [
      {
        name: 'cellCount',
        maximum: 2147483648,
        reason: 'Published sort ceiling and exact integer ranks.',
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
    limitation:
      'Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions remain 128.',
    limits: [
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-dequantize.ts'
      }
    ]
  },
  GPURasterEmbeddingQuantize: {
    limitation:
      'Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions remain 128.',
    limits: [
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-quantize.ts'
      }
    ]
  },
  GPURasterEmbeddingSimilarity: {
    limitation:
      'Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions remain 128.',
    limits: [
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
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
    limitation:
      'Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions remain 128.',
    limits: [
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
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
    limitation:
      'Default searchStructure tiled performs exact exhaustive training in 256-row tiles. Brute-force retains 4,096 training rows. Training count times dimensions must fit uint32 addressing, device bindings and the command budget; aligned whole-pixel chunks support 1–1,024 dimensions.',
    limits: [
      {
        name: 'trainingCount (brute-force)',
        maximum: 4096,
        reason: 'Private reference training traversal.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts'
      },
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts'
      },
      {
        name: 'trainingCount',
        maximum: 4294967295,
        reason:
          'Uint32 addressing; actual device storage bindings and backing allocations impose lower limits.',
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
    limitation:
      'Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions remain 128.',
    limits: [
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-linear-probe.ts'
      },
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
    limitation:
      'Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions remain 128.',
    limits: [
      {
        name: 'dimensions',
        maximum: 1024,
        reason:
          'Streaming bounded component traversal; all records and outputs must fit device bindings.',
        source:
          'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection.ts'
      },
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
    limitation:
      'Exactly one of vectors, quantized or aggregation is supplied. Chain aggregation.vectorSums and aggregation.validCount from output.vectorSums and output.validCount to preserve original-pixel support, including cancelling children. Vector validity requires nonempty, nonzero direction; sums/counts remain usable when direction is invalid. Caller-supplied aggregation counts must total at most 2^32 - 1 per output footprint (uint32); scale and dimensions also constrain each pass.',
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
    limitation:
      'Default storage-backed QR supports 4,096 slices with bounded scratch batches; mode reference retains 256 slices. Aligned cell chunks retain complete time-series records.',
    limits: [
      {
        name: 'sliceCount (reference)',
        maximum: 256,
        reason: 'Private reference QR arrays.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression.ts'
      },
      {
        name: 'sliceCount',
        maximum: 4096,
        reason: 'Storage-backed QR with bounded scratch batches.',
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
    limitation:
      'Default auto selects exact SCC through 1024 nodes and 65,536 CSR slots, parallel SCC above them; parallel resilience uses an auxiliary forest. Explicit exact retains the reference caps. SCC budget exhaustion is reported separately from completed resilience outputs; all bindings and auxiliary counts must fit.',
    tier: 'scale',
    limits: [
      {
        name: 'nodeCount',
        maximum: 2147483646,
        reason:
          'Parallel uint32 node addressing; device bindings and resilience auxiliary counts impose lower limits.',
        source: 'modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts'
      },
      {
        name: 'edgeCount',
        maximum: 2147483646,
        reason:
          'Parallel CSR slots are device-binding bound; resilience auxiliary counts impose an additional limit.',
        source: 'modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts'
      },
      {
        name: 'nodeCount (exact)',
        maximum: 1024,
        reason: 'Explicit bounded reference route.',
        source: 'modules/analysis/src/gpu-network/network-structure/gpu-network-structure.ts'
      },
      {
        name: 'edgeCount (exact)',
        maximum: 65536,
        reason: 'Explicit bounded reference route.',
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
        reason:
          'MAXIMUM_EVENT_COUNT: exhaustive all-pairs of events; applies to every correction mode.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'radiusCount',
        maximum: 256,
        reason: 'MAXIMUM_RADIUS_COUNT: per-radius accumulators; applies to every correction mode.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'pairCount',
        maximum: 1048576,
        reason:
          'MAXIMUM_PAIR_COUNT: eventCount squared must also fit the caller maximumPairCount budget in every correction mode.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'nodeCount',
        maximum: 65536,
        reason:
          'MAXIMUM_NODE_COUNT: bounds node-distance storage and shell scans in every correction mode.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'edgeCount',
        maximum: 65536,
        reason:
          'MAXIMUM_EDGE_COUNT: bounds edgeLengths rows and shell scans in every correction mode.',
        source: 'modules/analysis/src/gpu-network/gpu-network-inhomogeneous-k.ts'
      },
      {
        name: 'eventCount * nodeCount',
        maximum: 16777216,
        reason: 'MAXIMUM_DISTANCE_COUNT: bounds nodeDistances rows in every correction mode.',
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
    limitation:
      'Legacy solver selection and successive-shortest-path defaults remain. Opt-in mode primal-dual uses three phases by default, with bounded relaxation/phase budgets; incomplete feasible flows require completion and cut-certification status checks. Device bindings and command budgets constrain all routes.',
    tier: 'scale',
    limits: [
      {
        name: 'nodeCount',
        maximum: 30000000,
        reason:
          'Parallel solver is device-binding bound, up to 30,000,000 nodes; command budgets also apply.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-flow.ts'
      },
      {
        name: 'edgeCount',
        maximum: 30000000,
        reason:
          'Parallel solver is device-binding bound, up to 30,000,000 edges; command budgets also apply.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-flow.ts'
      },
      {
        name: 'nodeCount (reference)',
        maximum: 128,
        reason: 'Explicit bounded reference route.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-flow.ts'
      },
      {
        name: 'edgeCount (reference)',
        maximum: 1024,
        reason: 'Explicit bounded reference route.',
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
    tier: 'scale',
    limitation:
      'Supply topology for sparse parallel multilevel moving, device-binding bound up to 1,073,741,823 nodes and CSR slots, with a 65,536-command budget. The largest scratch binding needs 16 bytes per node or slot (8,388,608 at 128 MiB). Legacy dense weights retain the 32-node reference route.',
    limits: [
      {
        name: 'nodeCount',
        maximum: 32,
        reason: 'The vertex-moving passes run in one invocation over a dense weight matrix.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-louvain.ts'
      },
      {
        name: 'nodeCount (CSR)',
        maximum: 1073741823,
        reason:
          'Four strength words per node use uint32 addressing; device storage bindings further constrain capacity.',
        source: 'modules/analysis/src/gpu-network/network-optimization/network-louvain-sparse.ts'
      },
      {
        name: 'CSR edge capacity',
        maximum: 1073741823,
        reason:
          'Two two-word weak-neighbor records per CSR slot use uint32 addressing; device bindings further constrain capacity.',
        source: 'modules/analysis/src/gpu-network/network-optimization/network-louvain-sparse.ts'
      },
      {
        name: 'estimatedCommandNodeCount (CSR)',
        maximum: 65536,
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
    limitation:
      'Default mode scalable uses CSR frontier relaxation; explicit reference retains private per-query arrays. Predecessors require queries times edges storage. Scalable default maximumRelaxations is 64 (up to 1024); inspect iterationLimitReached before using incomplete routes.',
    tier: 'scale',
    limits: [
      {
        name: 'edgeCount',
        maximum: 4294967294,
        reason:
          'Default scalable CSR route is device-binding bound; edge IDs reserve the invalid uint32 sentinel.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-point-routing.ts'
      },
      {
        name: 'turnCount',
        maximum: 4294967295,
        reason: 'Default scalable CSR turn slots are device-binding bound and uint32 indexed.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-point-routing.ts'
      },
      {
        name: 'edgeCount (reference)',
        maximum: 128,
        reason: 'Explicit bounded reference route.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-point-routing.ts'
      },
      {
        name: 'turnCount (reference)',
        maximum: 4096,
        reason: 'Explicit bounded reference route.',
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
    tier: 'scale',
    limitation:
      'Parallel reductions are the default; mode reference retains the serial per-zone oracle. Lorenz knots and output bindings retain their existing limits.',
    limits: [
      {
        name: 'rows',
        maximum: 2147483648,
        reason: 'Published sort ceiling applies to parallel and reference routes.',
        source: 'modules/analysis/src/dataframe-analysis/composite-indicators/gpu-inequality.ts'
      },
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
    limitation:
      'Auto selects fixed Boruvka for larger fixed-count requests, supporting 65,535 regions subject to device bindings. Dense and adaptive routes retain 256 regions.',
    tier: 'scale',
    limits: [
      {
        name: 'regionCount (dense/adaptive)',
        maximum: 256,
        reason: 'Dense or adaptive reference envelope.',
        source: 'modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph.ts'
      },
      {
        name: 'regionCount',
        maximum: 65535,
        reason: 'Fixed-count Boruvka route; device bindings also apply.',
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
    limitation:
      'Default searchStructure bounds indexes zone pixel bounds and tiles candidate scratch; brute-force retains its dense pair cap. Candidate-budget exhaustion publishes saturated requiredCount and invalidates unvisited support; inspect candidateOverflow and output overflow. Identity transforms are preserved; nonidentity snapping is capped at 1e-5 pixel per axis (four float32 steps at most).',
    limits: [
      {
        name: 'zoneCountTimesCellCount (searchStructure brute-force)',
        maximum: 4194304,
        reason: 'Explicit dense reference coverage bounds zone by cell scratch.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts'
      },
      {
        name: 'candidateCapacity (bounds)',
        maximum: 4294967294,
        reason:
          'uint32 candidate offsets reserve a saturated sentinel; bindings and tiled command budgets apply.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts'
      }
    ]
  },
  GPURasterFractionalZonalStatistics: {
    limits: [
      {
        name: 'zoneCount',
        maximum: 4294967295,
        reason:
          'uint32 zone count; the 16-byte per-zone summary and all coverage, raster and output views must fit device bindings.',
        source: 'modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts'
      }
    ],
    limitation:
      'Consumes precomputed coverage with no inherited zone-times-cell cap. Uniform area uses planar coordinate units squared; geographic/web-mercator modes use ground square meters and require an axis-aligned grid. Explicit positive finite cellAreas require uniform cellSizeMode and exactly grid.width * grid.height rows, and cannot be combined with ground modes. Invalid support or unrepresentable aggregates suppress affected zones with NaNs and validity zero. Coverage construction defaults to bounds-indexed tiled candidates; only explicit brute-force coverage has the 4,194,304 pair ceiling. The 16-byte zone summary permits at most 8,388,608 zones at a 128 MiB binding limit.'
  },
  GPUTerrainBreach: {
    limitation:
      'Scratch and caller bindings are validated before graph declarations. elevationDecrement must round to a finite float32 value.',
    tier: 'reference',
    limits: [
      {
        name: 'scratchWords',
        maximum: 4294967295,
        reason:
          'Search scratch is pits times (steps + 1) times the search window, two words per cell, and must be addressable with 32 bits and fit the device storage binding.',
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
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      },
      {
        name: 'queriesTimesPositionsWithoutGrid',
        maximum: 134217728,
        reason: 'Without a grid every query tests every point.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudNormalScale: {
    limitation:
      'Minimum population plane roughness over supplied radii, excluding the query itself. Each scale uses four compute stages plus parameter uploads, with at most 32 scales. Truncated scales are rejected; grid coverage must include intended support.',
    limits: [
      {
        name: 'radii.length',
        maximum: 32,
        reason: 'Three graph kernels per scale.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      },
      {
        name: 'capacity',
        maximum: 64,
        reason: 'Per-query neighbor arrays.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      },
      {
        name: 'queriesTimesPositionsWithoutGrid',
        maximum: 134217728,
        reason: 'Exhaustive search without a grid.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudChange: {
    limits: [
      {
        name: 'samplingCovariancePoints',
        maximum: 32768,
        reason: 'One thread visits every pair of in-cylinder points, quadratic per core.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      },
      {
        name: 'coresTimesPointsWithoutGrid',
        maximum: 134217728,
        reason: 'Without a grid every core scans both clouds.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudICP: {
    limits: [
      {
        name: 'iterations',
        maximum: 100,
        reason: 'Iterations are unrolled into the command graph.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      },
      {
        name: 'sourceTimesTargetWithoutGrid',
        maximum: 134217728,
        reason: 'Without a target grid every source point tests every target point.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudSMRF: {
    limits: [
      {
        name: 'windowCells',
        maximum: 64,
        reason: 'Morphology window radius in cells; cost grows with the window.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      },
      {
        name: 'cutCells',
        maximum: 32,
        reason: 'Cut radius in cells.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      }
    ]
  },
  GPUPointCloudGround: {
    limits: [
      {
        name: 'radiusCells',
        maximum: 32,
        reason: 'Largest morphology radius in cells.',
        source: 'modules/analysis/src/pointcloud-analysis/operations/pointcloud-limits.ts'
      }
    ]
  },
  GPUFuzzyContiguity: {
    limitation:
      'BVH candidates and stable lexicographic pair ordering; required pair totals and packed word counts remain uint32-bounded. Device storage bindings constrain features, geometry and candidate output below numeric ceilings.',
    tier: 'scale',
    limits: [
      {
        name: 'features',
        maximum: 4294967294,
        reason:
          'Stable two-column pair ordering removes the packed 16-bit key cap; terminal feature offsets and device bindings constrain rows.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts'
      },
      {
        name: 'vertices',
        maximum: 2147483647,
        reason:
          'Two scalar position words per vertex use uint32 addressing; geometry and bounds must fit device bindings.',
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
    limitation:
      'Auto selects tiled unit effects when entity eigenvalues are supplied. Tiled fits support 1,024 entities and 16,777,216 total rows, subject to device bindings; reference fits retain 32 entities and 32 periods.',
    tier: 'reference',
    limits: [
      {
        name: 'rows (tiled)',
        maximum: 16777216,
        reason: 'Exact float32 count envelope.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts'
      },
      {
        name: 'entities (tiled)',
        maximum: 1024,
        reason: 'Supplied entity spectrum.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts'
      },
      {
        name: 'entities (reference)',
        maximum: 32,
        reason: 'A dense entity by entity determinant is formed per evaluation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts'
      },
      {
        name: 'periodCount (reference)',
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
    tier: 'scale',
    limitation:
      'Planar finite-support kernels index automatically above 65,536 rows unless disabled; bisquare or explicit Gaussian cutoff supports the larger indexed envelope. Dense and AICc selection remain reference work. Unpenalized GLM; separation is a sufficient float32 recession-direction certificate, not an exhaustive LP test. AICc needs a positive small-sample denominator.',
    limits: [
      {
        name: 'response rows (dense)',
        maximum: 65536,
        reason: 'Dense local IRLS scans a bounded training neighborhood per invocation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts'
      },
      {
        name: 'response rows (indexed finite-support)',
        maximum: 16777216,
        reason:
          'Finite-support indexed local fits are device-binding bound, up to 16,777,216 rows; exact unbounded Gaussian retains the legacy indexed ceiling.',
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
      },
      {
        name: 'response rows (indexed unbounded Gaussian)',
        maximum: 1048576,
        reason:
          'Legacy dense Gaussian scan with an explicit index; finite-support routes use the larger ceiling.',
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
    tier: 'scale',
    limits: [
      {
        name: "rows (neighborIndex 'brute-force')",
        maximum: 4096,
        reason: 'Every visit scans every row in the sequential exhaustive reference.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts'
      },
      {
        name: "rows (neighborIndex 'grid', the default above 4096)",
        maximum: 1048576,
        reason:
          'Grid DBSCAN is device-binding bound, up to 1,048,576 rows; default sequential frontier batches 256 visits. Explicit parallel ordering has a separate command budget.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts'
      },
      {
        name: 'rows (xi extraction)',
        maximum: 65536,
        reason: 'Separate xi reference cap; extraction batches 64 ranks.',
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
    limitation:
      'Aligned GraphVectorView query support, drift, value, status and weight records support chunked queries; complete support groups stay together and shared samples remain packed. One sample grid is reused across query chunks.',
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
        maximum: 2147483648,
        reason:
          'Recursive 256-row prefix tree; uint32 sort addressing allows up to 2,147,483,648 cells, further constrained by device bindings.',
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
    limitation:
      'Sorted ranks above 64 rows support up to 65,535 logical states, subject to dense transition-output bindings. Explicit pairwise ranking and transition diagnostics retain 64.',
    tier: 'scale',
    limits: [
      {
        name: 'rowCount',
        maximum: 65535,
        reason: 'Sorted rank route; the dense state matrix must also fit device bindings.',
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
    limitation:
      'periodCount is at most 2,147,483,647; cube and correlation address products must fit uint32.',
    limits: [
      {
        name: 'periodCount',
        maximum: 2147483647,
        reason: 'Signed lag and uint32 flattened address arithmetic.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation.ts'
      },
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
  GPUNestedInterpolationValidation: {
    limitation:
      'Default mode tiled supports 4,096 rows; mode reference retains 256 rows and 32 neighbors. Both retain fold, candidate and device limits.',
    tier: 'scale',
    limits: [
      {
        name: 'rows (reference)',
        maximum: 256,
        reason: 'Reference fold traversal.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts'
      },
      {
        name: 'rows',
        maximum: 4096,
        reason: 'Tiled folds and metrics.',
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
      'Cell and histogram populations are uint32, subject to device bindings/backing allocations and combined matching/reflected population. Quantized distributions and explicit input/reference domains remain; CLAHE omits the global output rescale.',
    limits: [
      {
        name: 'grid cell count (including reflected tile padding)',
        maximum: 4294967295,
        reason:
          'Uint32 addressing; actual device storage bindings and backing allocations impose lower limits.',
        source:
          'modules/analysis/src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts'
      },
      {
        name: 'histogramLength',
        maximum: 4294967295,
        reason:
          'Uint32 addressing; actual device storage bindings and backing allocations impose lower limits.',
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
  },
  GPUDorlingCartogram: {
    limitation:
      'Default neighborGrid auto selects the indexed route from 32,768 rows; device-binding bound, up to 1,073,741,822 rows. The largest plane is 16 times (rows + 1) bytes (8,388,607 rows at 128 MiB). Explicit neighborGrid false retains all-pairs. Heuristic relaxation does not guarantee overlap removal.',
    limits: [
      {
        name: 'rows (indexed)',
        maximum: 1073741822,
        reason: 'uint32 four-word addressing, further constrained by device storage bindings.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts'
      },
      {
        name: 'rows (all-pairs)',
        maximum: 65536,
        reason: 'Explicit exhaustive reference route.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts'
      },
      {
        name: 'maximumIterations',
        maximum: 256,
        reason: 'Bounded unrolled relaxation schedule.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts'
      }
    ]
  },
  GPUIslandAttachment: {
    limitation:
      'Default auto uses exhaustive scan through 4096 rows and a neighbor grid above it. Grid rows and CSR output slots are device-binding bound; explicit scan retains its 65,535-row reference ceiling.',
    limits: [
      {
        name: 'positions.length (grid)',
        maximum: 4294967294,
        reason:
          'uint32 row IDs reserve the invalid sentinel; device bindings further constrain capacity.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts'
      },
      {
        name: 'positions.length (scan)',
        maximum: 65535,
        reason: 'Explicit exhaustive scan reference.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts'
      }
    ]
  },
  GPUFlowAggregation: {
    limitation:
      'Stable two-column pair ordering above 65,535 zones; device bindings constrain zones and pair capacity. Chunked GraphVectorView origins, destinations and aligned weights are supported; shared zone geometry remains resident.',
    limits: [
      {
        name: 'zoneCount',
        maximum: 4294967294,
        reason:
          'uint32 zone IDs reserve the invalid sentinel; terminal offsets and device bindings apply.',
        source: 'modules/analysis/src/gpu-network/flow-aggregation/flow-aggregation-pairs.ts'
      }
    ]
  },
  GPUNetworkCoarsening: {
    limitation:
      'Dense group outputs and scratch are device-binding bound; lexicographic superedge sorting removes the former packed group-key ceiling.',
    limits: [
      {
        name: 'groupCapacity',
        maximum: 429496729,
        reason:
          'Ten scratch words per group use uint32 addressing; device bindings impose lower limits.',
        source: 'modules/analysis/src/gpu-network/network-coarsening/gpu-network-coarsening.ts'
      }
    ]
  },
  GPUGeographicallyWeightedRegression: {
    limitation:
      'Indexed finite-support fits are device-binding bound up to 16,777,216 rows. Gaussian defaults remain exact through 1,048,576 rows; above that count auto uses cutoff 8 and publishes approximation. Explicit gaussianMode exact retains the dense indexed cap; planar compact support indexes automatically.',
    limits: [
      {
        name: 'indexedRows',
        maximum: 16777216,
        reason: 'Indexed row IDs, selection scratch and complete bindings must fit the device.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters.ts'
      },
      {
        name: 'denseIndexedRows',
        maximum: 1048576,
        reason:
          'Exact untruncated Gaussian and unindexed nonstationarity fits retain the historical dense envelope.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters.ts'
      },
      {
        name: 'denseRows',
        maximum: 65536,
        reason: 'GWR without a grid retains the exhaustive reference ceiling.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters.ts'
      }
    ]
  },
  GPUGeographicallyWeightedRegressionNonstationarityTest: {
    limitation:
      'Indexed finite-support fits are device-binding bound up to 16,777,216 rows. Gaussian defaults remain exact through 1,048,576 rows; above that count auto uses cutoff 8 and publishes approximation. Explicit gaussianMode exact retains the dense indexed cap; planar compact support indexes automatically.',
    limits: [
      {
        name: 'indexedRows',
        maximum: 16777216,
        reason: 'Indexed row IDs, selection scratch and complete bindings must fit the device.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters.ts'
      },
      {
        name: 'denseIndexedRows',
        maximum: 1048576,
        reason:
          'Exact untruncated Gaussian and unindexed nonstationarity fits retain the historical dense envelope.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters.ts'
      }
    ]
  },
  GPUInverseDistanceWeighting: {
    limitation:
      'Capacities through 64 use private top-k; larger maximumNeighborCount uses storage-backed selection in device-sized output-cell batches. Zero (default) selects radius-only interpolation. All sample, cell and scratch bindings must fit the device.',
    limits: [
      {
        name: 'maximumNeighborCount',
        maximum: 16777216,
        reason: 'Exact float32 k parameter; storage-backed top-k removes the private 64-slot cap.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts'
      }
    ]
  },
  GPUNearestFeatureJoin: {
    limitation:
      'Private top-k through 64 slots, storage-backed top-k above 64; total slots remain uint32-bounded and device storage bindings impose lower limits. Nearest-feature point queries and aligned outputs accept chunked GraphVectorView input; neighbors mode retains its own packed geometry and slot layout. Optional neighborRows are original feature rows before sourceIds remapping; they require disjoint neighborIds output. Wire neighborRows into GPUPairGather.neighbors.rows; neighborIds may contain caller identities.',
    limits: [
      {
        name: 'k/neighborCapacity',
        maximum: 2147483646,
        reason:
          'Signed-compatible storage-backed top-k slots; rows times slots and device bindings also constrain capacity.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-join/nearest-types.ts'
      }
    ]
  },
  GPUNearestFeatureWeights: {
    limitation:
      'Private top-k through 64 slots, storage-backed top-k above 64; total slots remain uint32-bounded and device storage bindings impose lower limits. Input neighborRows are feature row indices, never application-valued neighborIds.',
    limits: [
      {
        name: 'slotCapacity',
        maximum: 2147483646,
        reason:
          'Signed-compatible storage-backed top-k slots; rows times slots and device bindings also constrain capacity.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-join/nearest-types.ts'
      }
    ]
  },
  GPUPairGather: {
    limitation:
      'Dense neighbor input is neighbors.rows (original feature row indices), not neighbors.ids or application IDs. candidateOverflow reports inherited candidate-stage loss separately from final-capacity overflow. rowCount is clamped to output capacity; optional status.matchedCount counts matched rows. Supplied candidate loss requires an output candidateOverflow destination.'
  },
  GPURasterFractionalCategoricalStatistics: {
    limitation:
      'Default tiled sort merges category support before normalization; tuple tiles retain at most 1,048,576 rows each. Explicit mode exact retains the original full-sort reduction and caps. Device bindings and command budgets constrain all routes.',
    limits: [
      {
        name: 'tuples (tiled)',
        maximum: 4294967294,
        reason: 'uint32 tuple indices reserve a saturated count sentinel.',
        source:
          'modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts'
      },
      {
        name: 'zoneCount * categoryCapacity (tiled)',
        maximum: 4294967294,
        reason:
          'uint32 group keys reserve the inactive sentinel; dense outputs must fit device bindings.',
        source:
          'modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts'
      },
      {
        name: 'tuples (exact)',
        maximum: 1048576,
        reason: 'Explicit full-sort reference.',
        source:
          'modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts'
      },
      {
        name: 'zoneCount * categoryCapacity (exact)',
        maximum: 65535,
        reason: 'Explicit dense group reference.',
        source:
          'modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts'
      },
      {
        name: 'categoryCapacity',
        maximum: 65534,
        reason: 'Dense category scratch and per-zone winner scan.',
        source:
          'modules/analysis/src/raster-analysis/gpu-raster-fractional-categorical-statistics.ts'
      }
    ]
  },
  addBuildingTessellationRecipe: {
    tier: 'reference',
    limitation:
      'Local mode uses an adaptive candidate index and publishes convex-piece triangle unions for bounded simple nonconvex or holed enclosures (at most 32 rings). Reference mode retains dissolved rings, 64 buildings and 128 samples. Device bindings and certification/overflow status still apply.',
    limits: [
      {
        name: 'buildingCount (local)',
        maximum: 1000000,
        reason:
          'Indexed local cell representation; sample, candidate, vertex and device binding limits also apply.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-tessellation-local.ts'
      },
      {
        name: 'buildingCount (reference)',
        maximum: 64,
        reason: 'Explicit global reference tessellation.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-tessellation.ts'
      },
      {
        name: 'sampleCapacity (reference)',
        maximum: 128,
        reason: 'Reference private clipping support.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-tessellation.ts'
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:addVisibilityGraphRecipe': {
    tier: 'reference',
    limitation:
      'CSR metrics use budgeted exact all-source calculations or deterministic sampled estimates. Auto mode samples when the exact command budget cannot fit; explicit exact mode rejects an insufficient budget. Sampled maximum depth is a lower bound.'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addVisibilityGraphRecipe': {
    tier: 'reference',
    limitation:
      'CSR mode provides visibility adjacency and degree only. Dense reference mode provides all-pairs metrics through 256 viewpoints. Network metrics for CSR are supplied by the separate gpu-network recipe.',
    limits: [
      {
        name: 'positions.length (csr)',
        maximum: 1000000,
        reason: 'Indexed sparse point queries; device bindings and grid visit budgets also apply.',
        source: 'modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr.ts'
      },
      {
        name: 'edgeCapacity (csr)',
        maximum: 16777216,
        reason: 'Sparse output slots and device scratch bindings.',
        source: 'modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr.ts'
      },
      {
        name: 'positions.length (reference)',
        maximum: 256,
        reason: 'Dense all-pairs reference.',
        source: 'modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph.ts'
      }
    ]
  },
  GPUFittingCoordinateEncoder: {
    limitation:
      'Aligned GraphVectorView coordinate, mask and encoded feature records support chunked rows; encoder parameters and frequencies remain shared.'
  },
  GPUCalendarBuckets: {
    limitation:
      'Aligned GraphVectorView timestamps, offsets, masks and calendar outputs support chunked rows; time-zone tables and parameters remain shared. Local day magnitude is at most 1,000,000,000; UTC offsets are within +/-1440 minutes or +/-86400 table seconds. Int64 offset addition must not overflow; invalid rows publish sentinels.'
  },
  GPUTimeWordsRelative: {
    limitation:
      'Aligned GraphVectorView timestamp and output rows support chunked input with a shared epoch and temporal context.'
  },
  GPUPeriodIndex: {
    limitation:
      'Aligned GraphVectorView timestamp or calendar columns, masks and period outputs support chunked rows; invalid counts merge across chunks.'
  },
  GPUCellPeriodKeys: {
    limitation:
      'Aligned GraphVectorView cell, period, mask and output key rows support chunked input.'
  },
  GPUClassAssignment: {
    limitation:
      'Aligned GraphVectorView observation, mask and output rows support chunked input; break tables remain shared.'
  },
  GPUAffineTransform: {
    limitation:
      'GraphVectorView vertices and aligned output support chunked input only without computed origins (origins: []); centroid and center origins retain the packed route.'
  },
  GPUDatumGridTransform: {
    limitation:
      'Aligned GraphVectorView positions, optional low parts and output rows support chunked input; the correction grid and sample mask remain packed and resident.'
  },
  GPUPointBuffer: {
    limitation:
      'GraphVectorView points and output offsets support chunked input; each fixed-size ring record stays whole.'
  },
  GPUCellGeometry: {
    limitation:
      'GraphVectorView cell rows and aligned output chunks are supported; complete fixed-stride geometry records stay within one chunk.'
  },
  GPUCellMeasures: {
    limitation:
      'GraphVectorView cell rows and aligned measure outputs support chunked input; complete edge-length records stay within one chunk.'
  },
  GPUCellGridPath: {
    limitation:
      'Supports H3, Quadbin and S2 paths, with exact integer S2 edge distance through level 30. Aligned chunked input retains complete fixed-stride output paths.',
    limits: [
      {
        name: 'S2 radius',
        maximum: 4294967294,
        reason: 'Reserves the undefined-distance sentinel.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path.ts'
      }
    ]
  },
  GPUCellTopology: {
    limitation:
      'GraphVectorView cell rows, masks and aligned outputs support chunked input; complete fixed-stride topology records stay within one chunk.'
  },
  GPUH3CellIdentities: {
    limitation:
      'GraphVectorView cell rows and aligned identity output chunks are supported; all six directed-edge slots for a cell stay together.'
  },
  GPUS2CellHierarchy: {
    limitation:
      'GraphVectorView cell rows and aligned hierarchy outputs support chunked input; complete fixed-stride child records stay within one chunk.'
  },
  GPUS2CellMeasures: {
    limitation:
      'GraphVectorView cell rows and aligned measure and status outputs support chunked input.'
  },
  GPUHealpixPointToCell: {
    limitation:
      'Aligned GraphVectorView position, mask, cell and validity rows support chunked input.'
  },
  GPUGaussianPredictiveScores: {
    limitation:
      'Aligned GraphVectorView mean, variance, observation and score rows support chunked input.'
  },
  GPULineLocate: {
    limitation:
      'GraphVectorView event rows and aligned outputs support chunked queries; global line geometry, offsets and the shared line-prefix stage remain resident.'
  },
  GPULinearReferencing: {
    limitation:
      'GraphVectorView query rows and aligned output chunks are supported; global line geometry, offsets and the shared prefix/index stage remain resident.'
  },
  GPULocalRBF: {
    limitation:
      'GraphVectorView queries and aligned value/status outputs support chunked queries; shared samples and supplied candidate lists remain packed.'
  },
  GPUNaturalNeighbor: {
    limitation:
      'GraphVectorView queries, optional triangle seeds and aligned outputs support chunked queries; shared triangulation and observations remain packed.'
  },
  GPUSphericalConvexContains: {
    limitation:
      'Aligned GraphVectorView queries, polygon selectors and output rows support chunked queries; polygon geometry remains packed and shared.'
  },
  GPUSphericalPolygonContains: {
    limitation:
      'Aligned GraphVectorView queries, polygon selectors and output rows support chunked queries; polygon geometry remains packed and shared.'
  },
  GPUPointToCell: {
    limitation:
      'GraphVectorView positions, aligned masks and output rows support chunked input; cell parameters remain shared.'
  },
  GPUPointInPolygonJoin: {
    limitation:
      'Chunked queries keep global source order and shared polygon geometry. candidateTileCapacity bounds paged candidate scratch independently of the unchanged global candidateCapacity; global overflow/status semantics remain.'
  },
  GPUBufferSelection: {
    limitation:
      'GraphVectorView points, source IDs, masks and aligned distances support chunked queries; shared feature geometry remains resident and compact output retains global source-row order. output.nearestFeatureRows always contains original feature rows, independent of feature sourceIds. The matches compact port exposes candidateOverflow separately from final-capacity overflow; top-level diagnostics retain their documented meanings.'
  },
  GPURegionMask: {
    limitation:
      'GraphVectorView positions and aligned mask output support chunked point queries; region geometry remains shared.'
  },
  GPURegionStatistics: {
    limitation:
      'GraphVectorView point positions and value rows support chunked input; shared region geometry and compact global summaries remain resident.'
  },
  GPUPointDensity: {
    limitation:
      'Chunked positions and weights use fixed tile scratch for packed hexagon/masked routes, controlled by scratchTileSize. Shared grid outputs remain packed; fixed-point scale and counts span all chunks.'
  },
  GPUKernelDensity: {
    limitation:
      'Explicit method binned accepts aligned chunked positions and weights. Corrected-weight scalar scratch splits when a global binding cannot fit; split totals use fixed-point accumulation. Exact and auto retain packed indexed inputs, and shared grid outputs must fit.'
  },
  GPUBoundsFilter: {
    limitation:
      'Chunked feature bounds use chunk-local bit masks and global block-count scratch. Compact IDs retain global source-row order; output bindings and global block counts must fit the device.'
  },
  GPUZonalStatistics: {
    limitation:
      'GraphVectorView positions, values and weights support chunked input; sorted sums require packed views, so chunked inputs default to atomic sums. Global zone tables and shared geometry remain resident.'
  },
  GPUCellAggregation: {
    limitation:
      'Aligned chunked inputs use fixed tile sorting scratch controlled by scratchTileSize. Aggregate padded tile rows are at most 4,294,967,295; packed global aggregate outputs remain device-binding bounded.',
    limits: [
      {
        name: 'aggregate padded tile rows',
        maximum: 4294967295,
        reason: 'Uint32 tile-union addressing.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts'
      }
    ]
  },
  GPUGroupStatistics: {
    limitation:
      'Reducible statistics use tiled scratch controlled by scratchTileSize. Compact tiled padded rows are at most 4,294,967,295; packed output and exact order-statistic scratch remain device-binding bounded.',
    limits: [
      {
        name: 'compact aggregate padded tile rows',
        maximum: 4294967295,
        reason: 'Uint32 tile-union addressing.',
        source: 'modules/analysis/src/dataframe-analysis/group-statistics/gpu-group-statistics.ts'
      }
    ]
  },
  GPUColumnQuantiles: {
    limitation:
      'GraphVectorView values and aligned masks support chunked input; histogram ranks merge across chunks. Rational probabilities without filter outputs allow up to 2,147,483,647 logical rows; float32 filter/rank routes retain 16,777,216.'
  },
  GPUClassBreaks: {
    limitation:
      'GraphVectorView values and aligned masks support chunked classification for compiled methods excluding maximum-breaks. Exact and monotone natural-breaks remain packed-only; global reductions and moment scratch preserve their existing row limits.'
  },
  GPUTransitionMatrix: {
    limitation:
      'Up to 65,535 classes subject to dense matrix and device binding limits; transition diagnostics retain 64 classes.',
    limits: [
      {
        name: 'classCount',
        maximum: 65535,
        reason: 'Dense matrix addressing and device bindings bound the class count.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix.ts'
      }
    ]
  },
  GPUSpatialInteraction: {
    limitation:
      'Dense routes retain 4,096 pairs and 128 groups. Sparse doubly constrained balancing accepts explicit pair tables, subject to device bindings and the command-node budget; incomplete balancing is reported.'
  },
  GPUColumnProfile: {
    limitation:
      'Chunked GraphVectorView columns support up to 4,294,967,295 logical rows, subject to device bindings. Counts are uint32; float32 statistical fields remain rounded.',
    limits: [
      {
        name: 'rows',
        maximum: 4294967295,
        reason:
          'Uint32 addressing; actual device storage bindings and backing allocations impose lower limits.',
        source: 'modules/analysis/src/dataframe-analysis/column-profile/gpu-column-profile.ts'
      }
    ]
  },
  GPUVariableStandardization: {
    limitation:
      'Chunked GraphVectorView columns support up to 4,294,967,295 logical rows, subject to device bindings. Counts are uint32; float32 statistical fields remain rounded.',
    limits: [
      {
        name: 'rows',
        maximum: 4294967295,
        reason:
          'Uint32 addressing; actual device storage bindings and backing allocations impose lower limits.',
        source:
          'modules/analysis/src/dataframe-analysis/variable-standardization/gpu-variable-standardization.ts'
      }
    ]
  },
  GPUCellCompaction: {
    limitation:
      'Supports H3, Quadbin and S2. S2 levels are 0–30; compaction expansion depth is at most 15. S2 covering uses level-15 center containment, rather than exact polygon intersection.',
    limits: [
      {
        name: 'S2 expansion depth',
        maximum: 15,
        reason: 'Bounds descendant expansion.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction.ts'
      },
      {
        name: 'S2 level',
        maximum: 30,
        reason: 'S2 uint64 cell layout.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction.ts'
      }
    ]
  },
  GPUCellCover: {
    limitation:
      'Supports H3, Quadbin and S2. S2 center containment refines only through level 15; it is not exact spherical polygon intersection.',
    limits: [
      {
        name: 'S2 feature count',
        maximum: 715827882,
        reason: 'Six initial faces per feature.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-cover/gpu-cell-cover.ts'
      },
      {
        name: 'S2 frontier count',
        maximum: 1073741823,
        reason: 'Four-word frontier records.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-cover/gpu-cell-cover.ts'
      },
      {
        name: 'S2 level',
        maximum: 15,
        reason: 'Bounded center-containment refinement.',
        source: 'modules/analysis/src/gpu-spatial-analysis/cell-cover/gpu-cell-cover.ts'
      }
    ]
  },
  GPUCellSetOutline: {
    limitation:
      'Supports H3, Quadbin and S2 outlines; S2 levels are 0–30. Fine-level geometry remains float32 and device/uint32 output bounds apply.'
  },
  GPUMapMatching: {
    limitation:
      'Map-matching row capacity is min(floor(bindingBytes / 12), 16,777,215). Optional status ports report validity, candidate overflow and iteration limits separately.',
    limits: [
      {
        name: 'rows',
        maximum: 16777215,
        reason: 'Float32 row identities and the three-word binding envelope.',
        source: 'modules/analysis/src/gpu-network/map-matching/gpu-map-matching.ts'
      }
    ]
  },
  GPURasterStandardizedIndex: {
    limitation:
      'Evaluation sliceCount and chunkSliceCount support 16,777,216; per-cell fitting limits and device bindings still apply. Complete cell time-series records may be chunked.',
    limits: [
      {
        name: 'chunkSliceCount',
        maximum: 16777216,
        reason: 'Exact float32 slice ranks; actual device bindings also apply.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-standardized-index.ts'
      },
      {
        name: 'sliceCount',
        maximum: 16777216,
        reason: 'Exact float32 slice ranks; actual device bindings also apply.',
        source:
          'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-standardized-index.ts'
      }
    ]
  },
  GPUSeeminglyUnrelatedRegression: {
    limitation:
      'Tiled moments and reductions support 16,777,216 total rows, subject to device bindings. Panel period count is bounded by the total row count.',
    limits: [
      {
        name: 'rows',
        maximum: 16777216,
        reason: 'Tiled reductions; float32 count envelope.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression.ts'
      }
    ]
  },
  GPUSpatialPanelErrorMaximumLikelihood: {
    limitation:
      'Tiled moments and reductions support 16,777,216 total rows, subject to device bindings. Panel period count is bounded by the total row count.',
    limits: [
      {
        name: 'periodCount',
        maximum: 16777216,
        reason: 'Total rows must also be at most 16,777,216.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood.ts'
      },
      {
        name: 'rows',
        maximum: 16777216,
        reason: 'Tiled reductions; float32 count envelope.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood.ts'
      }
    ]
  },
  GPUEnsembleScores: {
    limitation:
      'Variogram pair count is at most 2,147,483,647 to prevent flattened-index overflow; all score inputs and outputs remain device-binding bounded.',
    limits: [
      {
        name: 'variogramPairCount',
        maximum: 2147483647,
        reason: 'Flattened pair addressing.',
        source: 'modules/analysis/src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.ts'
      }
    ]
  },
  GPUTerrainTiledFlow: {
    tier: 'scale',
    limitation:
      'execution auto selects resident or bounded D8 tile/boundary reconciliation; explicit tiled requires D8 and resident retains the reference route. Tiled D8 keeps maximum-tile scratch plus a coarse boundary forest; caller inputs/outputs remain resident. No host paging or branching reconciliation.',
    limits: [
      {
        name: 'estimatedCommandNodeCount',
        maximum: 32768,
        reason: 'Bounded graph expansion.',
        source: 'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts'
      },
      {
        name: 'grid cell count',
        maximum: 536870911,
        reason: 'Eight fraction words per cell require uint32 addressing.',
        source: 'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts'
      }
    ]
  },
  GPUTerrainBranchingFlow: {
    limitation:
      'Branching flow remains resident and is bounded by 32,768 command nodes and actual device bindings.',
    limits: [
      {
        name: 'estimatedCommandNodeCount',
        maximum: 32768,
        reason: 'Bounded graph expansion.',
        source: 'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts'
      }
    ]
  },
  GPUTrajectoryEncounters: {
    limitation:
      'Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually impose much lower limits.',
    limits: [
      {
        name: 'hitCapacity',
        maximum: 4294967294,
        reason: 'Reserves 0xffffffff as the saturated overflow sentinel.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-encounters.ts'
      }
    ]
  },
  GPUTrajectoryClosestApproach: {
    limitation:
      'Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually impose much lower limits.',
    limits: [
      {
        name: 'hitCapacity',
        maximum: 4294967294,
        reason: 'Reserves 0xffffffff as the saturated overflow sentinel.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach.ts'
      },
      {
        name: 'entryCapacity',
        maximum: 4294967294,
        reason: 'Reserves 0xffffffff as the saturated overflow sentinel.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach.ts'
      }
    ]
  },
  GPUZoneEvents: {
    limitation:
      'Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually impose much lower limits.',
    limits: [
      {
        name: 'candidateCapacity',
        maximum: 4294967294,
        reason: 'Reserves 0xffffffff as the saturated overflow sentinel.',
        source: 'modules/analysis/src/gpu-spatial-analysis/trajectory-zones/gpu-zone-events.ts'
      }
    ]
  },
  GPUTrackSimilarity: {
    limitation:
      'invalidPair also reports invalid track ranges and densification overflow; an invalid pair must not be interpreted as a valid similarity.'
  },
  GPUSpaceTimeHawkes: {
    limitation:
      'On GPU, squared initialSpatialStandardDeviation must be finite normal float32 and at least 2^-126. The CPU reference retains its wider variance domain.'
  },
  GPURoseStatistic: {
    limitation:
      'Supports radians/degrees and positive-x counterclockwise or positive-y clockwise conventions. Rows and group validation retain uint32; groupCount times summary stride and binCount must each fit uint32 and device bindings.',
    limits: [
      {
        name: 'output elements',
        maximum: 4294967295,
        reason: 'Uint32 output addressing.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts'
      },
      {
        name: 'groupCount',
        maximum: 4294967295,
        reason: 'Also bounded by groupCount times summary stride and circular.binCount.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts'
      },
      {
        name: 'rows',
        maximum: 4294967295,
        reason: 'Uint32 input indexing.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts'
      }
    ]
  },
  GPUArealInterpolation: {
    limitation:
      'targetZoneCount times categories.categoryCount is at most 4,294,967,294, reserving the final scan sentinel; device bindings impose lower limits.',
    limits: [
      {
        name: 'targetZoneCount * categories.categoryCount',
        maximum: 4294967294,
        reason: 'Reserves the final category-share scan sentinel.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation.ts'
      }
    ]
  },
  GPUPointCloudHeight: {
    limitation:
      'Height queries and aligned outputs accept chunked GraphVectorView input. Height sampling settings are runtime inputs; grid axes and float32 support bounds are validated at construction.'
  },
  GPUSpatialPredicateJoin: {
    limitation:
      'candidateTileCapacity bounds paged candidate scratch independently of the unchanged global candidateCapacity. Final pairs and weights remain globally capacity bounded; inspect candidate and output overflow separately.'
  },
  GPUDotDensity: {
    limits: [
      {
        name: 'normalized feature count',
        maximum: 16777215,
        reason: 'Bounds the 256-invocation per-feature normalization dispatch.',
        source: 'modules/analysis/src/gpu-spatial-analysis/dot-density/gpu-dot-density.ts'
      }
    ]
  },
  GPUConstrainedDelaunayTriangulation: {
    limits: [
      {
        name: 'triangleCount (parallel)',
        maximum: 1431655765,
        reason: 'Three-word half-edge records.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts'
      },
      {
        name: 'polygonCount (parallel)',
        maximum: 536870911,
        reason: 'Eight-word records.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts'
      },
      {
        name: 'maximumFlips (serial)',
        maximum: 4294967294,
        reason: 'Reserves a final verification increment.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts'
      }
    ]
  },
  GPUDelaunayTessellation: {
    limits: [
      {
        name: 'output triangle count',
        maximum: 1431655765,
        reason: 'Three-word output records.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.ts'
      },
      {
        name: 'scratch triangle count',
        maximum: 1073741823,
        reason: 'Four-word scratch records.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.ts'
      }
    ]
  },
  GPUConcaveHull: {
    limits: [
      {
        name: 'triangle reservation',
        maximum: 715827882,
        reason: 'Six boundary words per triangle.',
        source: 'modules/analysis/src/gpu-spatial-analysis/concave-hull/gpu-concave-hull.ts'
      }
    ]
  },
  GPUGroupConcaveHull: {
    limits: [
      {
        name: 'triangle reservation',
        maximum: 715827882,
        reason: 'Six boundary words per triangle; total reservation must also fit device bindings.',
        source: 'modules/analysis/src/gpu-spatial-analysis/concave-hull/gpu-concave-hull.ts'
      }
    ],
    limitation:
      'For n input positions, T=max(0,2*n-5). Fixed layout reserves groupCount*T triangles and three edges per triangle; packed layout reserves T triangles total and three edges per triangle, with separate compact lists and groupCount triangle/edge offset/count rows. Both layouts require groupCount convergence rows and suppress nonconverged groups. Packed count/requiredCount count boundary edges; overflow is zero with the required full reservation.'
  },
  GPUVoronoiDiagram: {
    limits: [
      {
        name: 'segmentCount',
        maximum: 1073741823,
        reason: 'Four endpoint words per segment.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram.ts'
      },
      {
        name: 'triangleCount',
        maximum: 357913941,
        reason: 'Twelve candidate endpoint words per triangle.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram.ts'
      }
    ]
  },
  GPULabelPoint: {
    limits: [
      {
        name: 'gridSize',
        maximum: 65535,
        reason: 'Squared uint32 cell indices.',
        source: 'modules/analysis/src/gpu-spatial-analysis/label-point/gpu-label-point.ts'
      },
      {
        name: 'featureCount',
        maximum: 67108863,
        reason: '64-lane feature dispatch.',
        source: 'modules/analysis/src/gpu-spatial-analysis/label-point/gpu-label-point.ts'
      }
    ]
  },
  GPUShapeDescriptors: {
    limits: [
      {
        name: 'moment tile count',
        maximum: 4294967295,
        reason: 'Uint32 tile offsets.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors.ts'
      }
    ]
  },
  GPUSegmentSnap: {
    limitation:
      'Default grid search reuses candidate scratch in batches of at most 256 lines, with candidateCapacity default 1,024. Overflow uses bounded exhaustive fallback or reports incomplete geometry; candidateOverflowPolicy incomplete disables fallback. Explicit brute-force and fallback work are bounded by 16,777,216.',
    limits: [
      {
        name: 'exhaustive work',
        maximum: 16777216,
        reason: 'Bounds (positions + lines) times (references + 1)^3.',
        source: 'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap.ts'
      },
      {
        name: 'candidateCapacity',
        maximum: 4294967290,
        reason: 'Reserves batch headers.',
        source: 'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap.ts'
      }
    ]
  },
  GPUPolygonOverlay: {
    limitation:
      'Optional featureSources publishes representative retained-boundary source row, operand and mixed flag; it does not enumerate all input contributors. Tolerance is at most 2^127 and positive shader values must survive float32 conversion.',
    limits: [
      {
        name: 'tolerance',
        maximum: 2 ** 127,
        reason: 'Bounds float32 tolerance arithmetic.',
        source: 'modules/analysis/src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay.ts'
      }
    ]
  },
  GPUBufferSurface: {
    limitation:
      'Optional featureSources publishes representative retained-boundary source row, operand and mixed flag; it does not enumerate all input contributors. Tolerance is at most 2^127 and positive shader values must survive float32 conversion.',
    limits: [
      {
        name: 'tolerance',
        maximum: 2 ** 127,
        reason: 'Bounds float32 tolerance arithmetic.',
        source: 'modules/analysis/src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface.ts'
      }
    ]
  },
  GPUSegmentRingAssembly: {
    limitation:
      'vertexTolerance must round to positive finite float32 (at least 2^-149); caller and scratch bindings are checked before graph declarations.'
  },
  GPUOutlineGeometry: {
    limitation:
      'Compact output layouts are opt-in; offsets and counts describe only emitted geometry. Position addressing is at most 2,147,483,647 and candidate/fragment word counts must fit uint32.',
    limits: [
      {
        name: 'position address count',
        maximum: 2147483647,
        reason: 'Two-component uint32 position addressing.',
        source: 'modules/analysis/src/gpu-spatial-analysis/outline-geometry/gpu-outline-geometry.ts'
      }
    ]
  },
  GPUOffsetCurve: {
    limitation:
      'Compact output layouts are opt-in; offsets and counts describe only emitted geometry. Position addressing is at most 2,147,483,647 and candidate/fragment word counts must fit uint32.',
    limits: [
      {
        name: 'position address count',
        maximum: 2147483647,
        reason: 'Two-component uint32 position addressing.',
        source: 'modules/analysis/src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve.ts'
      }
    ]
  },
  GPULineSmooth: {
    limitation:
      'Optional source-row and source-chainage metadata follows the emitted vertices. Metadata rows are at most 1,073,741,823 and position addresses at most 2,147,483,647.',
    limits: [
      {
        name: 'position address count',
        maximum: 2147483647,
        reason: 'Two-component uint32 position addressing.',
        source: 'modules/analysis/src/gpu-spatial-analysis/line-segmentize/gpu-line-smooth.ts'
      },
      {
        name: 'metadata rows',
        maximum: 1073741823,
        reason: 'Four-word metadata addressing.',
        source: 'modules/analysis/src/gpu-spatial-analysis/line-segmentize/gpu-line-smooth.ts'
      }
    ]
  },
  GPULocalMoran: {
    limitation:
      'Rows are uint32/binding bounded up to 4,294,963,199. Fixed moment counts use extended exact uint32 encoding through 4,294,967,295; float32 statistics remain rounded.',
    limits: [
      {
        name: 'fixedCount',
        maximum: 4294967295,
        reason: 'Extended integer count-word encoding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts'
      },
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts'
      }
    ]
  },
  GPULocalGeary: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-additional-spatial-statistics.ts'
      }
    ]
  },
  GPUHotSpotAnalysis: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-hot-spot-analysis.ts'
      }
    ]
  },
  GPUSpatialPearson: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-additional-spatial-statistics.ts'
      }
    ]
  },
  GPUGammaStatistic: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-additional-spatial-statistics.ts'
      }
    ]
  },
  GPUGlobalPermutationTest: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test.ts'
      }
    ]
  },
  GPULocalPermutationTest: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test.ts'
      }
    ]
  },
  GPUGlobalSpatialStatistics: {
    limitation:
      'Default tiled reductions support analytic Moran, Geary, General G, bivariate Moran and join counts. Exact uint32 population, island and black counts accompany float32 statistics. Reference mode retains its prior limits; all bindings must fit.',
    limits: [
      {
        name: 'aggregate slots',
        maximum: 2147483647,
        reason: 'Bounds doubled binary symmetry counts.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics.ts'
      },
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics.ts'
      }
    ]
  },
  GPULocalSpatialHeteroscedasticity: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability.ts'
      }
    ]
  },
  GPUMultivariateLocalGeary: {
    limitation:
      'Scalable rows are bounded by 4,294,963,199 and device bindings. Integer count outputs preserve exact populations; float32 statistics remain rounded. Global statistics and descriptive multivariate Geary default to tiled reductions, with explicit mode reference retaining its prior limits.',
    limits: [
      {
        name: 'rows (scalable)',
        maximum: 4294963199,
        reason: 'Uint32 count/index arithmetic reserves 4,096 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability.ts'
      }
    ]
  },
  GPUMultipleTestingAdjustment: {
    limitation:
      'Exact integer ranks support at most 2,147,483,648 rows, subject to published sort and device bindings.',
    limits: [
      {
        name: 'rows',
        maximum: 2147483648,
        reason: 'Published sort ceiling; exclusive implementation bound is 2^31 + 1.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment.ts'
      }
    ]
  },
  GPUNetworkCentralityScheduleStep: {
    limitation:
      'Reusable fixed-lane pivot batches accumulate GPU-resident node/edge totals. First submission requires reset nonzero; pad the last batch with 0xffffffff. Keep the original sample-count scale and retain per-pivot outputs before the next submission. Each batch still must converge within its round budget.'
  },
  GPUKeyJoin: {
    limitation:
      'Exact finite-value count-distinct joins are available. Right rows are at most 2,147,483,648, subject to published sort and device binding limits.',
    limits: [
      {
        name: 'right rows',
        maximum: 2147483648,
        reason: 'Published sort ceiling.',
        source: 'modules/analysis/src/dataframe-analysis/key-join/gpu-key-join.ts'
      }
    ]
  },
  GPUEmpiricalBayesRates: {
    limitation:
      'Integer population counts support up to 4,294,967,039 rows, subject to device bindings and dispatch padding.',
    limits: [
      {
        name: 'rows',
        maximum: 4294967039,
        reason: 'Uint32 counts reserve 256 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/rate-smoothing/gpu-empirical-bayes-rates.ts'
      }
    ]
  },
  GPUSpatialEmpiricalBayesRates: {
    limitation:
      'Integer population counts support up to 4,294,967,039 rows, subject to device bindings and dispatch padding.',
    limits: [
      {
        name: 'rows',
        maximum: 4294967039,
        reason: 'Uint32 counts reserve 256 rows for padding.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates.ts'
      }
    ]
  },
  GPUSpatialScanStatistic: {
    limitation:
      'Accepted count totals are at most 4,294,967,294; 0xffffffff is reserved for overflow. Invalid simulations are disabled and an optional diagnostic exposes count overflow.',
    limits: [
      {
        name: 'count total',
        maximum: 4294967294,
        reason: 'Reserves the saturation sentinel.',
        source:
          'modules/analysis/src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts'
      }
    ]
  }
};

/** Internal contributors have no public catalog identity and are audited separately. */
export const INTERNAL_OPERATOR_ANNOTATIONS: Readonly<Record<string, OperatorAnnotation>> = {
  GPUSnapRoundedNoding: {
    limitation:
      'Planar only. lines.positions rows, and 4 x (rows + intersectionCapacity), are at most the device segment-intersection vertex limit (getGPUSegmentIntersectionMaximumVertexCount); leafCapacity must be a power of two. Incomplete output sets candidateOverflow or overflow.'
  }
};
