// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/index';
import type {KernelAuditCase} from './kernel-audit-context';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

/** Kernel-audit cases of the pair-merge restriction (MOB-04) and the segmented top-k search. */
export const MOB04_PAIR_MERGE_TOP_K_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUTrajectoryRestriction',
    name: 'pairwise merge with pairs from GPU views and a count',
    build: ({graph, view}) =>
      graph.add(
        new GPUTrajectoryRestriction({
          positions: view('float32x2', 1024),
          times: view('float32', 1024),
          offsets: view('uint32', 129),
          parameters: view('float32', 7),
          pairs: {
            first: view('uint32', 4096),
            second: view('uint32', 4096),
            count: view('uint32', 1)
          },
          intervalsPerPair: 16,
          output: {
            intervals: view('float32x2', 65536),
            positions: view('float32x4', 65536),
            status: view('uint32', 65536),
            relationships: view('uint32x4', 4096)
          }
        })
      )
  },
  ...[
    {metric: 'frechet', spatialContext: undefined},
    {metric: 'longest-common-subsequence', spatialContext: GREAT_CIRCLE},
    {metric: 'edit-distance-on-real-sequence', spatialContext: GREAT_CIRCLE},
    {metric: 'edit-distance-with-real-penalty', spatialContext: GREAT_CIRCLE},
    {metric: 'hausdorff', spatialContext: GREAT_CIRCLE}
  ].map(
    ({metric, spatialContext}): KernelAuditCase => ({
      exportName: 'GPUTrackSimilaritySearch',
      name: `segmented top1024 ${metric} ${spatialContext ? 'great-circle' : 'planar'}`,
      build: ({graph, view}) => {
        const subsequence = metric === 'longest-common-subsequence';
        graph.add(
          new GPUTrackSimilaritySearch({
            spatialContext,
            metric: metric as 'frechet',
            linesA: {
              kind: 'lines',
              positions: view('float32x2', 1024),
              lineOffsets: view('uint32', 129)
            },
            pairA: view('uint32', 4096),
            pairB: view('uint32', 4096),
            threshold: view('float32', 1),
            survivorCapacity: 1024,
            topK: 1024,
            output: {
              distance: subsequence ? undefined : view('float32', 4096),
              lowerBound: subsequence ? undefined : view('float32', 4096),
              similarity: subsequence ? view('float32', 4096) : undefined,
              upperBound: subsequence ? view('float32', 4096) : undefined,
              matched: view('uint32', 4096),
              status: view('uint32', 4096),
              topKIndices: view('uint32', 131072),
              topKSummary: view('uint32x2', 128)
            }
          })
        );
      }
    })
  )
];
