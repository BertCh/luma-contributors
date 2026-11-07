// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runExample as runTrajectoryKalman} from './trajectory-kalman.example';
import {runExample as runTrajectoryRestriction} from './trajectory-restriction.example';
import {runExample as runTrackSimilaritySearch} from './track-similarity-search.example';
import {Buffer, type Device, type VertexFormat} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {runS3Lens, type S3CaptureSpec} from './s3-lens-run';

export type TrajectoryCoverageKind =
  | 'trajectory-kalman'
  | 'trajectory-restriction'
  | 'track-similarity-search';

export function runTrajectoryCoverage(device: Device, kind: TrajectoryCoverageKind) {
  const element =
    kind === 'trajectory-kalman'
      ? 'GPUTrajectoryKalman'
      : kind === 'trajectory-restriction'
        ? 'GPUTrajectoryRestriction'
        : 'GPUTrackSimilaritySearch';
  return runS3Lens(device, {
    scenarioId: `gpu-spatial-analysis.${kind}`,
    element,
    datasets: [`synthetic-${kind}-v1`],
    metadata: {
      coordinateModel: 'projected',
      crs: 'synthetic local Cartesian metres',
      temporalModel: 'relative seconds'
    },
    keySpaces: [{id: 'entry', label: 'Explicit track, observation or query row', indexSpace: '1d'}],
    choices:
      kind === 'track-similarity-search'
        ? {metric: 'dynamic-time-warping', threshold: 0}
        : {
            parameters:
              kind === 'trajectory-kalman'
                ? 'q=0, measurement variance=1, initial velocity variance=1, maximum gap=10s'
                : 'closed time [0,4], rectangle [-1,-1,1,1], maximum gap=5s'
          },
    async build(graph, uploads) {
      const captures: Record<string, S3CaptureSpec> = {};
      const input = <Format extends VertexFormat>(
        name: string,
        format: Format,
        values: Float32Array | Uint32Array
      ) => {
        const view = uploads.view(name, values, format);
        captures[name] = {view, role: 'input', keySpace: 'entry'};
        return view;
      };
      const output = <Format extends VertexFormat>(
        name: string,
        format: Format,
        length: number,
        meaning: string
      ) => {
        const view = createTransientView(
          graph,
          name,
          format,
          length,
          Buffer.STORAGE | Buffer.COPY_SRC
        );
        captures[name] = {view, role: 'output', keySpace: 'entry', meaning};
        return view;
      };
      if (kind === 'trajectory-kalman') {
        await runTrajectoryKalman(graph, {
          positions: input('positions', 'float32x2', Float32Array.of(0, 0, 2, 0)),
          times: input('times', 'float32', Float32Array.of(0, 1)),
          offsets: input('offsets', 'uint32', Uint32Array.of(0, 2)),
          parameters: input('parameters', 'float32', Float32Array.of(0, 1, 1, 10)),
          output: {
            filteredState: output(
              'filteredState',
              'float32x4',
              2,
              '[x,y,vx,vy] from the forward observations only'
            ),
            filteredCovariance: output(
              'filteredCovariance',
              'float32x4',
              2,
              'Shared per-axis position/velocity covariance matrix'
            ),
            smoothedState: output(
              'smoothedState',
              'float32x4',
              2,
              'Backward RTS estimate using both observations'
            ),
            smoothedCovariance: output(
              'smoothedCovariance',
              'float32x4',
              2,
              'Backward RTS position/velocity covariance matrix'
            ),
            status: output(
              'status',
              'uint32',
              2,
              'Reset, prediction-only, invalid and singular-smoothing bits'
            )
          }
        });
      } else if (kind === 'trajectory-restriction') {
        await runTrajectoryRestriction(graph, {
          positions: input('positions', 'float32x2', Float32Array.of(-2, 0, 2, 0, 0, 0)),
          times: input('times', 'float32', Float32Array.of(0, 4, 20)),
          offsets: input('offsets', 'uint32', Uint32Array.of(0, 2, 3)),
          parameters: input('parameters', 'float32', Float32Array.of(0, 4, -1, -1, 1, 1, 5)),
          output: {
            positions: output(
              'clippedPositions',
              'float32x4',
              3,
              'Source-segment clipped endpoints, NaN when absent'
            ),
            intervals: output(
              'intervals',
              'float32x2',
              3,
              'Closed entry/exit time per source segment'
            ),
            status: output(
              'status',
              'uint32',
              3,
              'Interval, singleton, invalid, gap and temporal-support bits'
            ),
            relationships: output(
              'relationships',
              'uint32x4',
              2,
              'Ever, always, support rows, unknown rows; 0xffffffff means unknown'
            )
          }
        });
      } else {
        await runTrackSimilaritySearch(graph, {
          metric: 'dynamic-time-warping',
          maximumFrechetVertexCount: 8,
          linesA: {
            kind: 'lines',
            positions: input('positionsA', 'float32x2', Float32Array.of(0, 0, 1, 0)),
            lineOffsets: input('offsetsA', 'uint32', Uint32Array.of(0, 2))
          },
          linesB: {
            kind: 'lines',
            positions: input('positionsB', 'float32x2', Float32Array.of(0, 0, 1, 0, 10, 0, 11, 0)),
            lineOffsets: input('offsetsB', 'uint32', Uint32Array.of(0, 2, 4))
          },
          pairA: input('pairA', 'uint32', Uint32Array.of(0, 0)),
          pairB: input('pairB', 'uint32', Uint32Array.of(0, 1)),
          threshold: input('threshold', 'float32', Float32Array.of(0)),
          output: {
            distance: output(
              'distance',
              'float32',
              2,
              'Exact evaluated DTW distance; NaN for pruned pairs'
            ),
            lowerBound: output('lowerBound', 'float32', 2, 'Admissible maximum endpoint distance'),
            matched: output('matched', 'uint32', 2, 'Inclusive threshold membership'),
            status: output('status', 'uint32', 2, 'Evaluated=0; safely pruned=1')
          }
        });
      }
      return captures;
    }
  });
}
