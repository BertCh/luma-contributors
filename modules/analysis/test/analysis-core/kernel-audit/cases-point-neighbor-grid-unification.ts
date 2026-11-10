// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import type {KernelAuditCase} from './kernel-audit-context';

const ROWS = 5000;
const QUERY_ROWS = 200;
const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

/**
 * Kernel-audit cases for the time axis of `GPUNeighborSearch` at maximal options: every optional
 * output, planar and great-circle, self and cross joins, unbucketed and bucketed. The cases without
 * times stay in `cases-cells-and-joins.ts`.
 */
export const POINT_NEIGHBOR_GRID_UNIFICATION_AUDIT_CASES: KernelAuditCase[] = (
  ['knn', 'radius'] as const
).flatMap(mode =>
  [undefined, {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const].flatMap(
    spatialContext =>
      [false, true].flatMap(cross =>
        [1, 8].map(
          (timeBucketCount): KernelAuditCase => ({
            exportName: 'GPUNeighborSearch',
            name: `times ${mode} ${spatialContext ? 'great-circle' : 'planar'} ${cross ? 'cross' : 'self'} join buckets ${timeBucketCount}`,
            build: ({graph, view}) => {
              const queryRows = cross ? QUERY_ROWS : ROWS;
              const slots = mode === 'knn' ? queryRows * GPU_NEIGHBOR_SEARCH_MAXIMUM_K : 4096;
              graph.add(
                new GPUNeighborSearch({
                  mode,
                  k: mode === 'knn' ? GPU_NEIGHBOR_SEARCH_MAXIMUM_K : undefined,
                  spatialContext,
                  positions: view('float32x2', ROWS),
                  times: view('float32', ROWS),
                  ...(cross
                    ? {
                        queryPositions: view('float32x2', QUERY_ROWS),
                        queryTimes: view('float32', QUERY_ROWS)
                      }
                    : {}),
                  temporalContext: TEMPORAL_CONTEXT,
                  timeBucketCount,
                  mask: view('uint32', ROWS),
                  queryMask: view('uint32', queryRows),
                  parameters: view('float32', GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH),
                  gridSize: [64, 64],
                  output: {
                    offsets: view('uint32', queryRows + 1),
                    neighbors: view('uint32', slots),
                    weights: view('float32', slots),
                    distances: view('float32', slots),
                    neighborCounts: view('uint32', queryRows),
                    status: {
                      overflow: view('uint32', 1),
                      requiredCount: view('uint32', 1),
                      invalidCount: view('uint32', 1)
                    }
                  }
                })
              );
            }
          })
        )
      )
  )
);
