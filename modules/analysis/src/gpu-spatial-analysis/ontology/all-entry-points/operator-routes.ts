// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../spatial-analysis-ontology';
import {GPU_SPATIAL_JOIN_PREPARED_ROUTES} from '../../spatial-join/spatial-join-prepared-planning';
import {GPU_SPATIAL_PREDICATE_JOIN_ROUTES} from '../../spatial-join/spatial-predicate-route-plan';
import {GPU_NEIGHBOR_SEARCH_ROUTES} from '../../neighbor-search/neighbor-search-planning';
import {GPU_RANK_MARKOV_ROUTES} from '../../distribution-dynamics/rank-markov-routes';
import {GPU_SPATIAL_INTERACTION_ROUTES} from '../../spatial-interaction/spatial-interaction-routes';
import {GPU_GEOGRAPHIC_DISTRIBUTION_ROUTES} from '../../geographic-distribution/geographic-distribution-routes';
import {GPU_CELL_PYRAMID_LIMITS} from '../../cell-aggregation/cell-pyramid-planning';

const SOURCE_ROOT = 'modules/analysis/src/';
const SPATIAL = '@luma-contributors/analysis/gpu-spatial-analysis:';
const DATAFRAME = '@luma-contributors/analysis/dataframe-analysis:';

/** Source-owned descriptors where available; descriptive rows never certify whole-request fit. */
export const OPERATOR_ROUTES: Readonly<Record<string, readonly GPUAnalysisRoute[]>> = {
  [`${SPATIAL}GPUSpatialJoinPrepared`]: GPU_SPATIAL_JOIN_PREPARED_ROUTES,
  [`${SPATIAL}GPURankMarkov`]: GPU_RANK_MARKOV_ROUTES,
  '@luma-contributors/analysis/raster-analysis:GPURasterRegionGraph': [
    {
      name: 'dense-output',
      kind: 'allocation',
      quantity: 'adjacency and weights (each)',
      unit: 'bytes',
      byteExpression: 'regionCount * regionCount * 4 <= device.maxStorageBufferBindingSize',
      appliesWhen: 'all search routes; sparse search does not remove dense output allocation',
      scope: 'binding',
      accuracyImplications: 'Tiling input does not lift this whole-output bound.',
      source: `${SOURCE_ROOT}raster-analysis/remote-sensing/gpu-raster-region-graph.ts`
    }
  ],
  [`${SPATIAL}GPUSpatialInteraction`]: GPU_SPATIAL_INTERACTION_ROUTES,
  [`${SPATIAL}GPUGeographicDistribution`]: GPU_GEOGRAPHIC_DISTRIBUTION_ROUTES,
  [`${SPATIAL}GPUSpatialPredicateJoin`]: GPU_SPATIAL_PREDICATE_JOIN_ROUTES,
  [`${SPATIAL}GPUNeighborSearch`]: GPU_NEIGHBOR_SEARCH_ROUTES,
  [`${SPATIAL}GPUCellPyramid`]: [
    ...GPU_CELL_PYRAMID_LIMITS,
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellPyramid'
      },
      name: 'level-count-slots',
      kind: 'allocation',
      quantity: 'level count scratch',
      unit: 'bytes',
      byteExpression:
        'resolutions.length * 64 * 4 <= min(device.maxBufferSize, device.maxStorageBufferBindingSize)',
      appliesWhen: 'output.levelCounts is requested',
      scope: 'binding',
      accuracyImplications: 'Counts only; other aggregation and scratch bindings must also fit.',
      source: `${SOURCE_ROOT}gpu-spatial-analysis/cell-aggregation/gpu-cell-pyramid.ts`
    }
  ],
  [`${SPATIAL}GPUPointDensity`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointDensity'
      },
      name: 'workgroup',
      kind: 'allocation',
      quantity: 'fixed-point cell sums',
      unit: 'bytes',
      byteExpression:
        'gridSize[0] * gridSize[1] * 12 <= min(134217728, device.maxStorageBufferBindingSize, device.maxBufferSize)',
      appliesWhen:
        'weighted sums/means requested by statistic or output ports, and sumAccumulation auto or workgroup; at most 11184810 cells in the whole grid, independent of input chunking',
      scope: 'binding',
      alternative: 'atomic',
      accuracyImplications:
        'Atomic float accumulation is order dependent; workgroup fixed-point sums are deterministic.',
      source: `${SOURCE_ROOT}gpu-spatial-analysis/point-density/gpu-point-density.ts`
    }
  ],
  [`${SPATIAL}GPUKernelDensity`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUKernelDensity'
      },
      name: 'grid-index',
      kind: 'allocation',
      quantity: 'index grid cells',
      unit: 'bytes',
      byteExpression:
        '(indexGridSize[0] * indexGridSize[1] + 1) * 4 <= device.maxStorageBufferBindingSize',
      appliesWhen:
        'all methods share indexGridSize admission independently of output gridSize; exact index scratch uses this offsets binding; binned additionally checks padded raster and convolution constraints',
      scope: 'binding',
      alternative: 'binned only for Gaussian kernels; shared indexGridSize admission still applies',
      accuracyImplications:
        'Binned approximates point placement and convolution; exact evaluates distances.',
      source: `${SOURCE_ROOT}gpu-spatial-analysis/point-density/gpu-kernel-density.ts`
    }
  ],
  [`${DATAFRAME}GPUClassBreaks`]: [
    {
      name: 'materialized',
      planner: {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUClassBreaks'
      },
      kind: 'hard-limit',
      quantity: 'values rows',
      unit: 'rows',
      maximum: 1024,
      appliesWhen:
        'natural-breaks enabled, naturalBreaksMode exact and naturalBreaksExactStorage materialized',
      scope: 'domain',
      alternative: 'recompute',
      accuracyImplications:
        'Recompute retains exact dynamic programming, using less storage and more work.',
      source: `${SOURCE_ROOT}dataframe-analysis/column-classification/natural-breaks-limits.ts`
    },
    {
      name: 'recompute',
      planner: {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUClassBreaks'
      },
      kind: 'hard-limit',
      quantity: 'values rows',
      unit: 'rows',
      maximum: 8192,
      appliesWhen:
        'natural-breaks enabled, naturalBreaksMode exact and storage auto, omitted or recompute',
      scope: 'domain',
      alternative: 'monotone or approximate',
      accuracyImplications: 'Monotone and binned approximate routes may change class boundaries.',
      source: `${SOURCE_ROOT}dataframe-analysis/column-classification/natural-breaks-limits.ts`
    }
  ],
  [`${SPATIAL}GPUExpectationBasedSpaceTimeScan`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUExpectationBasedSpaceTimeScan'
      },
      name: 'indexed-search',
      kind: 'auto-crossover',
      quantity: 'maximumNeighbors',
      unit: 'rows',
      maximum: 65,
      conditionExpression: 'maximumNeighbors < zones && maximumNeighbors <= 65',
      appliesWhen:
        'nearestZoneSearch auto selects indexed when maximumNeighbors < zones and maximumNeighbors <= 65, otherwise full-scan; this producer independently caps maximumNeighbors at 64',
      scope: 'domain',
      alternative: 'full-scan',
      accuracyImplications:
        'Both searches are exact, including ties; index sorting changes cost, not the selected closest zones.',
      source: `${SOURCE_ROOT}gpu-spatial-analysis/scan-statistics/scan-statistic-nearest-index.ts`
    },
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUExpectationBasedSpaceTimeScan'
      },
      name: 'neighborhood-sort',
      kind: 'hard-limit',
      quantity: 'maximumNeighbors',
      unit: 'rows',
      maximum: 64,
      appliesWhen: 'all nearestZoneSearch routes; maximumWindowLength also <= 32',
      scope: 'domain',
      accuracyImplications: 'Zone search does not remove bounded neighborhood/window constraints.',
      source: `${SOURCE_ROOT}gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan.ts`
    }
  ]
};
