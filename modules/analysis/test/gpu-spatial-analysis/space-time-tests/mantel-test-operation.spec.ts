// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {
  GPUMantelTestOperation,
  type MantelTestOperationProps
} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';

it('both Mantel operation forms carry pair loss ports and recover', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: MantelTestOperationProps = {
    rowCount: 3,
    maximumPermutations: 3,
    pairs: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      distances: 'distances',
      status: {overflow: 'loss', candidateOverflow: 'candidateLoss'}
    },
    times: 'times',
    parameters: 'parameters',
    output: {statistics: 'statistics', summary: 'summary'}
  };
  for (const Operation of [GPUMantelTestOperation, CPUMantelTestOperation]) {
    const settings: MantelTestOperationProps = {
      ...props,
      pairs: {...props.pairs, status: {...props.pairs.status}},
      output: {...props.output}
    };
    const operation = new Operation(settings);
    delete settings.pairs.status!.overflow;
    delete settings.pairs.status!.candidateOverflow;
    settings.output.summary = 'replacement';
    for (const [loss, candidateLoss] of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      const result = await runOperationGraph(device, {
        operations: [operation],
        outputs: ['summary'],
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 3, 3)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
          distances: {format: 'float32', values: Float32Array.of(1, 2, 4)},
          times: {format: 'float32', values: Float32Array.of(0, 1, 3)},
          parameters: {
            format: 'uint32',
            values: getGPUSpaceTimeParameterValues({seed: 17, permutationCount: 3})
          },
          loss: {format: 'uint32', values: Uint32Array.of(loss)},
          candidateLoss: {format: 'uint32', values: Uint32Array.of(candidateLoss)}
        }
      });
      if (loss || candidateLoss)
        expect(Array.from(result['summary']).every(Number.isNaN)).toBe(true);
      else expect(result['summary'][0]).toBeCloseTo(Math.sqrt(3 / 28), 5);
    }
  }
});
import {
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const constant of [false, true])
  for (const stopping of [false, true])
    for (const transformed of [false, true]) {
      it(`Mantel constant=${constant}, stopping=${stopping}, transformed=${transformed} compares all outputs`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const transform = {constant: 1, power: -1};
        const props: MantelTestOperationProps = {
          rowCount: 5,
          pairs: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            distances: 'distances'
          },
          times: 'times',
          parameters: 'parameters',
          maximumPermutations: 31,
          ...(stopping ? {stopAfterExceedances: 4} : {}),
          ...(transformed ? {distanceTransform: transform, timeTransform: transform} : {}),
          output: {
            statistics: 'statistics',
            summary: 'summary',
            ...(stopping ? {permutationCount: 'used'} : {})
          }
        };
        const inputs = {
          offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 4, 7, 9, 10, 10)},
          neighbors: {
            format: 'uint32' as const,
            values: Uint32Array.of(1, 2, 3, 4, 2, 3, 4, 3, 4, 4)
          },
          weights: {
            format: 'float32' as const,
            values: Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
          },
          distances: {
            format: 'float32' as const,
            values: Float32Array.of(1.3, 4.1, 2.7, 8.2, 6.3, 3.4, 1.8, 9.1, 7.2, 5.3)
          },
          times: {
            format: 'float32' as const,
            values: constant ? new Float32Array(5) : Float32Array.of(0, 2, 5, 11, 17)
          },
          parameters: {
            format: 'uint32' as const,
            values: getGPUSpaceTimeParameterValues({seed: 2 ** 40 + 123, permutationCount: 23})
          }
        };
        const outputs = ['statistics', 'summary', ...(stopping ? ['used'] : [])];
        const gpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new GPUMantelTestOperation(props)]
        });
        const cpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new CPUMantelTestOperation(props)]
        });
        const exactSummary = new Set<number>([
          GPU_SPACE_TIME_SUMMARY.pairCount,
          GPU_SPACE_TIME_SUMMARY.timeClosePairs,
          GPU_SPACE_TIME_SUMMARY.expected,
          GPU_SPACE_TIME_SUMMARY.greaterCount,
          GPU_SPACE_TIME_SUMMARY.lesserCount
        ]);
        for (const port of outputs)
          for (let index = 0; index < cpu[port].length; index++) {
            const expected = cpu[port][index];
            const actual = gpu[port][index];
            if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
            else if (
              port === 'used' ||
              (port === 'summary' && exactSummary.has(index)) ||
              expected === 0
            )
              expect(actual === expected).toBe(true);
            // Correlations/means/z scores: 1e-4 absolute; variance/positive p-values: 1e-4 relative.
            else if (
              port === 'summary' &&
              (index === GPU_SPACE_TIME_SUMMARY.permutationVariance ||
                (index >= GPU_SPACE_TIME_SUMMARY.pseudoPGreater &&
                  index <= GPU_SPACE_TIME_SUMMARY.pseudoPTwoSided))
            )
              expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-4 * Math.abs(expected));
            else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-4);
          }
      });
    }
