// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../spatial-analysis-ontology';
import {GPU_ISLAND_ATTACHMENT_ROUTES} from '../../spatial-weights/island-attachment-routes';
import {GPU_SPATIAL_WEIGHTS_TRACES_ROUTES} from '../../spatial-regression/spatial-weights-traces-routes';
const SPATIAL = '@luma-contributors/analysis/gpu-spatial-analysis:';
const ROOT = 'modules/analysis/src/';

/** Selection rules and coupled limits supplement the individual allocation descriptors. */
export const OPERATOR_ROUTE_DETAILS: Readonly<Record<string, readonly GPUAnalysisRoute[]>> = {
  [`${SPATIAL}GPUSpatialInteraction`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialInteraction'
      },
      name: 'auto-method',
      kind: 'auto-crossover',
      quantity: 'flow pairs and groups',
      unit: 'rows',
      conditionExpression:
        '(pairs > 4096 || originCount > 128 || destinationCount > 128) && constraint === doubly && calibration !== bounded',
      appliesWhen:
        'method auto selects sparse only under the condition; otherwise dense, whose limits can reject the request',
      scope: 'domain',
      alternative: 'explicit sparse requires doubly constrained grid calibration',
      accuracyImplications:
        'Sparse balancing honors iteration/convergence budgets; dense uses the reference solve.',
      source: `${ROOT}gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction.ts`
    }
  ],
  [`${SPATIAL}GPUCellPyramid`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellPyramid'
      },
      name: 'auto-scratch',
      kind: 'auto-crossover',
      quantity: 'first-level source rows',
      unit: 'rows',
      conditionExpression:
        'scratchMode !== packed && (scratchTileSize supplied ? rows > scratchTileSize : rows * 4 > min(device.maxStorageBufferBindingSize, device.maxBufferSize))',
      appliesWhen:
        'inherited cell aggregation scratchMode auto; each level still publishes a single packed table',
      scope: 'binding',
      alternative: 'packed; must fit all scratch bindings',
      accuracyImplications:
        'Both routes retain exact cell keys, counts and fixed-point sums. Coarser levels propagate finer-level capacity loss.',
      source: `${ROOT}gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts`
    }
  ],
  [`${SPATIAL}GPUPointDensity`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointDensity'
      },
      name: 'auto-sums',
      kind: 'auto-crossover',
      quantity: 'weighted sums',
      unit: 'elements',
      conditionExpression: 'sumAccumulation omitted or auto => workgroup',
      appliesWhen:
        'auto always selects deterministic workgroup sums; no size-triggered fallback to atomic',
      scope: 'domain',
      alternative: 'explicit atomic',
      accuracyImplications:
        'Atomic changes reproducibility and last float bits; large weighted grids can require this explicit opt-in.',
      source: `${ROOT}gpu-spatial-analysis/point-density/gpu-point-density.ts`
    }
  ],
  [`${SPATIAL}GPUKernelDensity`]: [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUKernelDensity'
      },
      name: 'auto-method',
      kind: 'auto-crossover',
      quantity: 'Gaussian support in raster cells',
      unit: 'elements',
      conditionExpression:
        'valid && support > f32(32 * minimumCell) && support <= f32(f32(2 * binnedMargin) * minimumCell)',
      appliesWhen:
        'method auto with Gaussian kernel: a per-frame GPU gate selects binned under the condition, otherwise exact; support = f32(5 * f32(bandwidth)), minimumCell = min(abs(f32(cellSize.x)), abs(f32(cellSize.y))); non-Gaussian auto resolves to exact; omitted method defaults to exact',
      scope: 'domain',
      alternative: 'explicit exact or Gaussian binned',
      accuracyImplications:
        'Binned splats points and uses a three-box Gaussian approximation; bandwidth and padding determine its numerical envelope. Host declarations cannot certify the per-frame Gaussian gate: without frame bandwidth and cell size the numeric plan remains auto. Shared index admission applies to all routes.',
      source: `${ROOT}gpu-spatial-analysis/point-density/kernel-density-binned.ts`
    }
  ],
  '@luma-contributors/analysis/dataframe-analysis:GPUClassBreaks': [
    {
      name: 'auto-storage',
      kind: 'auto-crossover',
      quantity: 'values rows',
      unit: 'rows',
      maximum: 1024,
      appliesWhen:
        'natural-breaks enabled, naturalBreaksMode exact and naturalBreaksExactStorage auto or omitted: materialized through 1024 rows, recompute above; enabled exact admission rejects above 8192',
      scope: 'domain',
      alternative: 'approximate or monotone (explicit)',
      accuracyImplications:
        'Both exact storage routes retain the exact dynamic program; approximate and monotone can change boundaries.',
      source: `${ROOT}dataframe-analysis/column-classification/gpu-class-breaks.ts`
    },
    {
      name: 'monotone',
      kind: 'hard-limit',
      quantity: 'values rows',
      unit: 'rows',
      maximum: 262144,
      appliesWhen: 'natural-breaks enabled and naturalBreaksMode monotone',
      scope: 'domain',
      alternative: 'approximate',
      accuracyImplications: 'Monotone is a constrained approximation, not full exact Fisher-Jenks.',
      source: `${ROOT}dataframe-analysis/column-classification/natural-breaks-limits.ts`
    }
  ],
  [`${SPATIAL}GPUIslandAttachment`]: [
    ...GPU_ISLAND_ATTACHMENT_ROUTES.map(route => ({
      ...route,
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis' as const,
        exportName: 'GPUIslandAttachment'
      }
    })),
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUIslandAttachment'
      },
      name: 'grid-candidates',
      kind: 'allocation',
      quantity: 'candidate slots',
      unit: 'elements',
      maximum: 4294967295,
      conditionExpression: 'rows * 16 <= 4294967295',
      appliesWhen:
        'nearestSearch resolves to grid; implies rows <= 268435455 before device and command limits',
      scope: 'domain',
      accuracyImplications: 'Arithmetic admission does not certify scratch or command fit.',
      source: `${ROOT}gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts`
    },
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUIslandAttachment'
      },
      name: 'grid-schedule',
      kind: 'hard-limit',
      quantity: 'commands',
      unit: 'commands',
      maximum: 32768,
      conditionExpression: '4096 + 3 * ceil(rows / 65536) * ceil(rows / 16777215) <= 32768',
      appliesWhen: 'grid exact-fallback schedule; independent of runtime unresolved count',
      scope: 'dispatch',
      accuracyImplications: 'Budget is a whole-declaration constraint, not per-chunk capacity.',
      source: `${ROOT}gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts`
    }
  ],
  [`${SPATIAL}GPUSpatialWeightsTraces`]: [
    ...GPU_SPATIAL_WEIGHTS_TRACES_ROUTES,
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialWeightsTraces'
      },
      name: 'forced-exact',
      kind: 'hard-limit',
      quantity: 'rows',
      unit: 'rows',
      maximum: 16384,
      appliesWhen: 'mode explicitly exact',
      scope: 'domain',
      alternative: 'stochastic',
      accuracyImplications:
        'Stochastic changes power traces from all unit probes to sampled probes.',
      source: `${ROOT}gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces.ts`
    }
  ],
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainTiledFlow': [
    {
      name: 'auto-resident',
      kind: 'auto-crossover',
      quantity: 'resident fraction plane',
      unit: 'bytes',
      byteExpression:
        'grid.width * grid.height * 32 <= min(device.maxStorageBufferBindingSize, device.maxBufferSize)',
      appliesWhen:
        'D8 execution auto runs resident when this plane fits, tiled otherwise; branching is resident and explicit tiled rejects branching',
      scope: 'binding',
      alternative: 'D8 tiled; explicit resident must fit',
      accuracyImplications:
        'Tiled uses wide boundary sums, resident float32 chains can accumulate more rounding error; maximumIterations is ignored by tiled D8.',
      source: `${ROOT}gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts`
    },
    {
      name: 'tiled-schedule',
      kind: 'hard-limit',
      quantity: 'commands',
      unit: 'commands',
      maximum: 32768,
      conditionExpression: '512 + 5 * tiles.length <= 32768',
      appliesWhen:
        'Only the selected tiled D8 route is charged this estimate; auto selects on the graph device. Resident admission uses estimatedCommandNodeCount for the selected resident route; worstCaseCommandNodeCount reports the maximum of both estimates only for automatic D8. For forced resident or forced tiled it equals that forced route estimate; branching is always resident and its worst-case estimate equals its resident estimate.',
      scope: 'dispatch',
      alternative: 'resident only when its domain, bindings and selected-route schedule fit',
      accuracyImplications:
        'For 6452 one-cell tiles and one sweep, selected resident admits 12932 commands; the separately named worst-case estimate is 32772. The whole-graph command-node budget is 32768 on every selected route; this is not a per-tile budget.',
      source: `${ROOT}gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts`
    }
  ],
  '@luma-contributors/analysis/raster-analysis:GPURasterFractionalCategoricalStatistics': [
    {
      name: 'zone-category-product',
      kind: 'hard-limit',
      quantity: 'zone/category groups including missing category',
      unit: 'elements',
      maximum: 4294967294,
      conditionExpression:
        'zoneCount * (categoryCapacity + 1) <= (mode === exact ? 65535 : 4294967294)',
      appliesWhen:
        'all modes reserve one missing-category slot; exact also caps tuples at 1048576 while tiled caps each tuple tile at 1048576',
      scope: 'domain',
      alternative: 'tiled (default), subject to output binding and schedule limits',
      accuracyImplications: 'Missing-category conservation is retained on both routes.',
      source: `${ROOT}raster-analysis/gpu-raster-fractional-categorical-statistics.ts`
    },
    {
      name: 'tuple-schedule',
      kind: 'hard-limit',
      quantity: 'commands',
      unit: 'commands',
      maximum: 32768,
      conditionExpression: '8 + 256 * ceil(tupleCount / tupleTileSize) <= 32768',
      appliesWhen: 'all modes; at most 127 tuple tiles before binding checks',
      scope: 'dispatch',
      accuracyImplications: 'Tiling input does not lift zone/category output bindings.',
      source: `${ROOT}raster-analysis/gpu-raster-fractional-categorical-statistics.ts`
    }
  ],
  '@luma-contributors/analysis/gpu-network:addVisibilityGraphRecipe': [
    {
      planner: {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPU_NETWORK_VISIBILITY_GRAPH_PLANNER'
      },
      name: 'pivot-metrics',
      kind: 'allocation',
      quantity: 'centrality lanes',
      unit: 'bytes',
      byteExpression:
        'laneCount * nodeCount * 12 <= min(134217728, device.maxStorageBufferBindingSize, device.maxBufferSize)',
      conditionExpression: 'pivotCount % laneCount === 0 && estimatedCommandNodeCount <= 32768',
      appliesWhen:
        'CSR network metrics; exact requires all-source pivot count. When pivotCount and pivotNodes are omitted, auto first tries all sources, then searches smaller feasible whole batches from its default sample. Explicit pivotCount or pivotNodes preserves the requested pivots and rejects infeasible work. Admission checks lane divisibility, allocation and binding bytes, composed adjacency and total recipe commands (at most 32768).',
      scope: 'binding',
      alternative:
        'inspect addVisibilityGraphRecipe.executionPlanner with CSR props and the graph before declaration',
      accuracyImplications:
        'Sampled betweenness is scaled and sampled maximum depth is a lower bound. Both visibility recipe identities retain their separate owners.',
      source: `${ROOT}gpu-network/visibility-graph/visibility-graph.ts`
    }
  ]
};
