// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 1_700_000_000_000n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};
const GEOGRAPHIC = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function compact({view}: KernelAuditContext, capacity: number) {
  return {
    ids: view('uint32', capacity, 'compact-ids'),
    status: {
      count: view('uint32', 1, 'compact-count'),
      overflow: view('uint32', 1, 'compact-overflow'),
      requiredCount: view('uint32', 1, 'compact-required')
    }
  };
}

/**
 * Kernel-audit cases for the time input of the DBSCAN and OPTICS contributors (ST-DBSCAN and
 * ST-OPTICS) at maximal options: every optional output wired, both sum orders, planar and
 * geographic positions. The cases without times stay in their original audit files.
 */
export const SPACE_TIME_CLUSTERING_AUDIT_CASES: KernelAuditCase[] = [
  ...(['atomic', 'sorted'] as const).flatMap(sumOrder =>
    [false, true].map(
      (geographic): KernelAuditCase => ({
        exportName: 'GPUSpatialClustering',
        name: `space-time every output ${geographic ? 'geographic' : 'planar'} ${sumOrder}`,
        build: context => {
          const {graph, view} = context;
          const rows = 5000;
          const capacity = 512;
          graph.add(
            new GPUSpatialClustering({
              positions: view('float32x2', rows),
              times: view('float32', rows),
              temporalContext: TEMPORAL_CONTEXT,
              parameters: view('float32', GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH),
              gridSize: [128, 128],
              sourceIds: view('uint32', rows),
              sumOrder,
              ...(geographic ? {spatialContext: GEOGRAPHIC} : {}),
              output: {
                labels: view('uint32', rows),
                rootRows: view('uint32', rows),
                coreFlags: view('uint32', rows),
                clusterCount: view('uint32', 1),
                clusters: compact(context, capacity),
                drawInstanceCount: view('uint32', 1),
                clusterSizes: view('uint32', capacity),
                clusterCentroids: view('float32x2', capacity)
              }
            })
          );
        }
      })
    )
  ),
  ...(['dbscan', 'xi'] as const).flatMap(extractionMethod =>
    [false, true].map(
      (geographic): KernelAuditCase => ({
        exportName: 'GPUOPTICS',
        name: `space-time 4096 rows,256 minimum samples,${extractionMethod} extraction,${geographic ? 'geographic' : 'planar'}`,
        build: ({graph, view}) => {
          graph.add(
            new GPUOPTICS({
              positions: view('float32x2', 4096),
              times: view('float32', 4096),
              temporalContext: TEMPORAL_CONTEXT,
              maximumTimeDistance: 60_000,
              minimumSamples: 256,
              maximumIterations: 4096,
              maximumDistance: 10,
              extractionMethod,
              ...(extractionMethod === 'dbscan' ? {extractionDistance: 10} : {}),
              ...(geographic ? {spatialContext: GEOGRAPHIC} : {}),
              output: {
                ordering: view('uint32', 4096),
                coreDistances: view('float32', 4096),
                reachability: view('float32', 4096),
                predecessors: view('uint32', 4096),
                labels: view('uint32', 4096),
                status: view('uint32', 3),
                ...(extractionMethod === 'xi'
                  ? {
                      clusterIntervals: view('uint32x2', 4096),
                      clusterSummary: view('uint32', 2)
                    }
                  : {})
              }
            })
          );
        }
      })
    )
  )
];
