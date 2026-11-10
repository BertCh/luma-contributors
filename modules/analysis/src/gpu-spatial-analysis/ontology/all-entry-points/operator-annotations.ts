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
        reason: 'The allocation runs in one invocation over all origins and facilities.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-rational-access.ts'
      },
      {
        name: 'supplyCount',
        maximum: 32,
        reason: 'The allocation runs in one invocation over all origins and facilities.',
        source:
          'modules/analysis/src/gpu-network/network-optimization/gpu-network-rational-access.ts'
      }
    ]
  },
  GPUNetworkLouvain: {
    tier: 'reference',
    limits: [
      {
        name: 'nodeCount',
        maximum: 32,
        reason: 'The vertex-moving passes run in one invocation over a dense weight matrix.',
        source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-louvain.ts'
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
        reason: 'Dense traversal of all pairs, O(rows squared times radii), in one invocation.',
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
        maximum: 16384,
        reason: 'Bounded dense traversal of all pairs, O(rows squared times bins).',
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
    limitation:
      'The correlated vertical-shift fit is serial and quadratic in the cell count (one invocation sums over all cell pairs), which is impractical beyond roughly 100,000 cells.'
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
        maximum: 512,
        reason: 'All feature pairs are tested exhaustively.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts'
      },
      {
        name: 'vertices',
        maximum: 65536,
        reason: 'Vertices are scanned for every pair.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-fuzzy-contiguity.ts'
      }
    ]
  },
  GPUInstrumentalVariablesCovariance: {
    limits: [
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
  GPUMultiscaleGeographicallyWeightedRegression: {
    limits: [
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
        name: 'rows',
        maximum: 4096,
        reason: 'Reachability is computed over all point pairs and the order is built serially.',
        source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts'
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
  GPUParameterBuffer: {
    role: 'helper'
  }
};
