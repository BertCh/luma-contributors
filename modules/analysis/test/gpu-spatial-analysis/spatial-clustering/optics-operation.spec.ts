// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUOPTICSOperation} from '../../../src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation';
import {
  GPUOPTICSOperation,
  type OPTICSOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const neighborIndex of ['brute-force', 'grid'] as const) {
  for (const extractionMethod of ['dbscan', 'xi'] as const) {
    for (const scene of ['ordinary', 'all-invalid', 'duplicates', 'temporal'] as const) {
      it(`OPTICS graph twins agree: ${neighborIndex}, ${extractionMethod}, ${scene}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const coordinates =
          scene === 'all-invalid'
            ? [NaN, Infinity, -Infinity]
            : scene === 'duplicates'
              ? [0, 0, 1, 10, 10, 11]
              : [0, 1, 3, 10, 11, 13, 50];
        const positions = Float32Array.from(coordinates.flatMap(value => [value, 0]));
        const rowCount = coordinates.length;
        const props: OPTICSOperationProps = {
          positions: 'positions',
          rowCount,
          minimumSamples: 2,
          maximumDistance: 5,
          extractionDistance: 2.5,
          extractionMethod,
          xi: 0.2,
          minimumClusterSize: 2,
          neighborIndex,
          neighborCapacity: neighborIndex === 'grid' ? rowCount * rowCount : undefined,
          ...(scene === 'temporal'
            ? {
                times: 'times',
                maximumTimeDistance: 2,
                temporalContext: {
                  encoding: 'float32-relative',
                  unit: 'second',
                  epoch: 0n,
                  axis: {kind: 'irregular'}
                } as const
              }
            : {}),
          output: {
            ordering: 'ordering',
            coreDistances: 'coreDistances',
            reachability: 'reachability',
            predecessors: 'predecessors',
            labels: 'labels',
            status: 'status',
            ...(extractionMethod === 'xi'
              ? {clusterIntervals: 'clusterIntervals', clusterSummary: 'clusterSummary'}
              : {}),
            ...(neighborIndex === 'grid'
              ? {ports: {status: {candidateOverflow: 'candidateOverflow'}}}
              : {})
          }
        };
        const outputs = [
          'ordering',
          'coreDistances',
          'reachability',
          'predecessors',
          'labels',
          'status',
          ...(extractionMethod === 'xi' ? ['clusterIntervals', 'clusterSummary'] : []),
          ...(neighborIndex === 'grid' ? ['candidateOverflow'] : [])
        ];
        const results = [];
        for (const Operation of [GPUOPTICSOperation, CPUOPTICSOperation]) {
          results.push(
            await runOperationGraph(device, {
              inputs: {
                positions: {format: 'float32x2', values: positions},
                ...(scene === 'temporal'
                  ? {
                      times: {
                        format: 'float32' as const,
                        values: Float32Array.from({length: rowCount}, (_, row) =>
                          row === rowCount - 1 ? NaN : row % 3
                        )
                      }
                    }
                  : {})
              },
              operations: [new Operation(props)],
              outputs
            })
          );
        }
        for (const port of outputs) {
          const actual = results[0][port];
          const expected = results[1][port];
          expect(actual.length).toBe(expected.length);
          for (let row = 0; row < expected.length; row++) {
            // Every discrete word and infinity is exact; finite distances allow 2e-5 relative
            // error with a 2e-5 absolute floor. Scenes avoid ambiguous rounded distance ties.
            if (
              (port === 'coreDistances' || port === 'reachability') &&
              Number.isFinite(expected[row])
            )
              expect(Math.abs(actual[row] - expected[row]), `${port}[${row}]`).toBeLessThanOrEqual(
                2e-5 * Math.max(1, Math.abs(expected[row]))
              );
            else expect(actual[row], `${port}[${row}]`).toBe(expected[row]);
          }
        }
      });
    }
  }
}
