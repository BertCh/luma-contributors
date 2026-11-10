// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPACE_TIME_CROSS_CORRELATION_MAXIMUM_LAGS,
  GPUSpaceTimeCrossCorrelation,
  GPUSpaceTimeLocalMoran,
  GPUSpatialWeightsAlgebra
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const SLOTS = 8;
const CELLS = 4096;

function buildSpaceTimeWeights(
  {graph, view, maximumRows}: KernelAuditContext,
  layout: 'period-major' | 'cell-major'
): void {
  // Output rows at the binding limit of the offsets (4 bytes per row).
  const periodCount = Math.floor(maximumRows(4) / CELLS) - 1;
  const rows = CELLS * periodCount;
  graph.add(
    new GPUSpatialWeightsAlgebra({
      operation: 'spaceTime',
      weights: {
        offsets: view('uint32', CELLS + 1),
        neighbors: view('uint32', CELLS * SLOTS),
        weights: view('float32', CELLS * SLOTS)
      },
      periodCount,
      layout,
      lags: [
        {lag: 0},
        {lag: 1, weight: 0.5, includeSelf: true},
        {lag: 2, weight: 0.25},
        {lag: -1, weight: 0.125, includeSelf: true}
      ],
      output: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', maximumRows(4)),
        weights: view('float32', maximumRows(4)),
        status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
      }
    })
  );
}

function buildLocalMoran(
  {graph, view, weights, maximumRows}: KernelAuditContext,
  variant: 'lagged' | 'differential',
  layout: 'period-major' | 'cell-major'
): void {
  const periodCount = Math.floor(maximumRows(4) / CELLS);
  graph.add(
    new GPUSpaceTimeLocalMoran({
      weights: weights(CELLS, SLOTS),
      cube: view('float32', CELLS * periodCount),
      layout,
      periodCount,
      period: periodCount - 1,
      lag: 2,
      variant,
      direction: 'outward',
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations: 999,
      maximumNeighbors: 32,
      islands: 'keep',
      multipleTesting: 'benjamini-yekutieli',
      output: {
        exceedances: view('uint32', CELLS),
        pseudoPValues: view('float32', CELLS),
        observed: view('float32', CELLS),
        significant: view('uint32', CELLS),
        adjustedPValues: view('float32', CELLS),
        islandCount: view('uint32', 1),
        quadrants: variant === 'lagged' ? view('uint32', CELLS) : undefined,
        status: {overflow: view('uint32', 1)}
      }
    })
  );
}

function buildCrossCorrelation(
  {graph, view, maximumRows}: KernelAuditContext,
  layout: 'period-major' | 'cell-major'
): void {
  const lagCount = GPU_SPACE_TIME_CROSS_CORRELATION_MAXIMUM_LAGS;
  const periodCount = 2 * lagCount + 8;
  // The cube is the larger binding (periods > lags); the outputs hold cells * lags.
  const cellCount = Math.floor(maximumRows(4) / periodCount);
  graph.add(
    new GPUSpaceTimeCrossCorrelation({
      first: view('float32', cellCount * periodCount),
      second: view('float32', cellCount * periodCount),
      layout,
      cellCount,
      periodCount,
      lags: Array.from({length: lagCount}, (_, index) => index - lagCount / 2),
      multipleTesting: 'benjamini-yekutieli',
      output: {
        correlations: view('float32', cellCount * lagCount),
        pValues: view('float32', cellCount * lagCount),
        adjustedPValues: view('float32', cellCount * lagCount),
        pairCounts: view('uint32', cellCount * lagCount),
        testedCount: view('uint32', 1)
      }
    })
  );
}

/** Cases of the space-time weights, local Moran recipes and cross-correlation (STX-02). */
export const SPACE_TIME_LOCAL_STATISTICS_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    name: 'spaceTime period-major four lags with self, binding-limited rows',
    build: context => buildSpaceTimeWeights(context, 'period-major')
  },
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    name: 'spaceTime cell-major four lags with self, binding-limited rows',
    build: context => buildSpaceTimeWeights(context, 'cell-major')
  },
  {
    exportName: 'GPUSpaceTimeLocalMoran',
    name: 'lagged outward period-major, binding-limited cube',
    build: context => buildLocalMoran(context, 'lagged', 'period-major')
  },
  {
    exportName: 'GPUSpaceTimeLocalMoran',
    name: 'differential cell-major, binding-limited cube',
    build: context => buildLocalMoran(context, 'differential', 'cell-major')
  },
  {
    exportName: 'GPUSpaceTimeCrossCorrelation',
    name: '64 lags period-major every output, binding-limited cells',
    build: context => buildCrossCorrelation(context, 'period-major')
  },
  {
    exportName: 'GPUSpaceTimeCrossCorrelation',
    name: '64 lags cell-major every output, binding-limited cells',
    build: context => buildCrossCorrelation(context, 'cell-major')
  }
];
