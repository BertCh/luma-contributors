// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUTrackSimilarity,
  GPUTrackSimilaritySearch
} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import type {KernelAuditCase} from './kernel-audit-context';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

/** Kernel-audit cases of the EDR, ERP and SSPD outputs and the bounded searches over them. */
export const TRACK_SIMILARITY_MEASURE_AUDIT_CASES: readonly KernelAuditCase[] = [
  ...[
    {name: 'planar', spatialContext: undefined},
    {name: 'great-circle', spatialContext: GREAT_CIRCLE}
  ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
    [
      {densify: 1 / 4096, maximumFrechetVertexCount: 2048},
      {densify: 1, maximumFrechetVertexCount: 1},
      {densify: undefined, maximumFrechetVertexCount: undefined}
    ].map(
      ({densify, maximumFrechetVertexCount}): KernelAuditCase => ({
        exportName: 'GPUTrackSimilarity',
        name: `edit measures ${name} densify ${densify} frechet ${maximumFrechetVertexCount} edr erp sspd`,
        build: ({graph, view, maximumRows, construct}) => {
          graph.add(
            construct(GPUTrackSimilarity, {
              spatialContext,
              linesA: {
                kind: 'lines',
                positions: view('float32x2', maximumRows(8)),
                lineOffsets: view('uint32', 4097)
              },
              pairA: view('uint32', 4096),
              pairB: view('uint32', 4096),
              activePairCount: view('uint32', 1),
              densify,
              maximumFrechetVertexCount,
              editDistanceTolerance: 250,
              editDistanceGapPoint: [-179.5, 89.5],
              output: {
                editDistanceOnRealSequence: view('float32', 4096),
                editDistanceWithRealPenalty: view('float32', 4096),
                symmetricSegmentPathDistance: view('float32', 4096),
                pairStatus: view('uint32', 4096)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUTrackSimilarity',
    name: 'edit measures at the pair maximum',
    build: ({graph, view, maximumRows, construct}) => {
      const pairs = 16777215;
      graph.add(
        construct(GPUTrackSimilarity, {
          linesA: {
            kind: 'lines',
            positions: view('float32x2', maximumRows(8)),
            lineOffsets: view('uint32', maximumRows(4))
          },
          pairA: view('uint32', pairs),
          pairB: view('uint32', pairs),
          maximumFrechetVertexCount: 2048,
          output: {
            editDistanceOnRealSequence: view('float32', pairs),
            editDistanceWithRealPenalty: view('float32', pairs),
            symmetricSegmentPathDistance: view('float32', pairs)
          }
        })
      );
    }
  },
  ...(
    [
      'hausdorff',
      'edit-distance-on-real-sequence',
      'edit-distance-with-real-penalty',
      'symmetric-segment-path'
    ] as const
  ).map(
    (metric): KernelAuditCase => ({
      exportName: 'GPUTrackSimilaritySearch',
      name: `${metric},2048 DP columns,binding-limited pair scratch,bounded top1024`,
      build: ({graph, view, maximumRows}) => {
        const pairCount = maximumRows(2048 * 4);
        graph.add(
          new GPUTrackSimilaritySearch({
            metric,
            maximumFrechetVertexCount: 2048,
            editDistanceTolerance: 1000,
            editDistanceGapPoint: [1000, -1000],
            survivorCapacity: 1024,
            topK: 1024,
            linesA: {
              kind: 'lines',
              positions: view('float32x2', 4096),
              lineOffsets: view('uint32', 129)
            },
            linesB: {
              kind: 'lines',
              positions: view('float32x2', 4096),
              lineOffsets: view('uint32', 3)
            },
            pairA: view('uint32', pairCount),
            pairB: view('uint32', pairCount),
            threshold: view('float32', 1),
            output: {
              distance: view('float32', pairCount),
              lowerBound: view('float32', pairCount),
              matched: view('uint32', pairCount),
              status: view('uint32', pairCount),
              topKIndices: view('uint32', 129 * 1024 - 1024),
              topKSummary: view('uint32x2', 128)
            }
          })
        );
      }
    })
  )
];
