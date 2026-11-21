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

for (const extractionMethod of ['dbscan', 'xi'] as const) {
  for (const maximumIterations of [1, 4]) {
    it(`X7 GPU and CPU OPTICS ${extractionMethod} preserve the ${maximumIterations}-visit budget`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const complete = maximumIterations === 4;
      const props: OPTICSOperationProps = {
        positions: 'positions',
        rowCount: 4,
        minimumSamples: 2,
        maximumDistance: 2,
        extractionDistance: 1.5,
        extractionMethod,
        maximumIterations,
        output: {
          ordering: 'ordering',
          coreDistances: 'coreDistances',
          reachability: 'reachability',
          predecessors: 'predecessors',
          labels: 'labels',
          ...(extractionMethod === 'xi'
            ? {clusterIntervals: 'clusterIntervals', clusterSummary: 'clusterSummary'}
            : {}),
          ports: {
            status: {
              iterationCount: 'iterationCount',
              invalidCount: 'invalidCount',
              converged: 'converged',
              iterationLimitReached: 'iterationLimitReached'
            }
          }
        }
      };
      const outputs = [
        'ordering',
        'labels',
        'invalidCount',
        'iterationCount',
        'converged',
        'iterationLimitReached',
        ...(extractionMethod === 'xi' ? ['clusterIntervals', 'clusterSummary'] : [])
      ];
      for (const Operation of [GPUOPTICSOperation, CPUOPTICSOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 4, 0, 5, 0)}
          },
          operations: [new Operation(props)],
          outputs
        });
        expect(Array.from(result['ordering'])).toEqual(
          complete ? [0, 1, 2, 3] : [0, 0xffffffff, 0xffffffff, 0xffffffff]
        );
        expect(Array.from(result['labels'])).toEqual(
          complete ? [0, 0, 1, 1] : [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]
        );
        expect(Array.from(result['invalidCount'])).toEqual([0]);
        expect(Array.from(result['iterationCount'])).toEqual([maximumIterations]);
        expect(Array.from(result['converged'])).toEqual([Number(complete)]);
        expect(Array.from(result['iterationLimitReached'])).toEqual([Number(!complete)]);
        if (!complete && extractionMethod === 'xi') {
          expect(Array.from(result['clusterSummary'])).toEqual([0, 0]);
          expect(Array.from(result['clusterIntervals'])).toEqual(new Array(8).fill(0xffffffff));
        }
      }
    });
  }
}

for (const {neighborIndex, orderingMethod} of [
  {neighborIndex: 'brute-force', orderingMethod: 'sequential'},
  {neighborIndex: 'grid', orderingMethod: 'sequential'},
  {neighborIndex: 'grid', orderingMethod: 'parallel'}
] as const) {
  for (const extractionMethod of ['dbscan', 'xi'] as const) {
    for (const scene of ['ordinary', 'all-invalid', 'duplicates', 'temporal'] as const) {
      it(`OPTICS graph twins agree: ${neighborIndex}, ${orderingMethod}, ${extractionMethod}, ${scene}`, async () => {
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
          orderingMethod,
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
            ...(extractionMethod === 'xi'
              ? {clusterIntervals: 'clusterIntervals', clusterSummary: 'clusterSummary'}
              : {}),
            ports: {
              status: {
                iterationCount: 'iterationCount',
                invalidCount: 'invalidCount',
                converged: 'converged',
                iterationLimitReached: 'iterationLimitReached',
                ...(neighborIndex === 'grid' ? {candidateOverflow: 'candidateOverflow'} : {})
              }
            }
          }
        };
        const outputs = [
          'ordering',
          'coreDistances',
          'reachability',
          'predecessors',
          'labels',
          'invalidCount',
          'iterationCount',
          'converged',
          'iterationLimitReached',
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
