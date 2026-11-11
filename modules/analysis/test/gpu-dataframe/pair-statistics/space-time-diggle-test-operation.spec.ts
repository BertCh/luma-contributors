// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpaceTimeDiggleTestOperation} from '../../../src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation';
import {
  GPUSpaceTimeDiggleTestOperation,
  type SpaceTimeDiggleTestOperationProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test-operation';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {
  getGPUSpaceTimeDiggleRandomParameterValues,
  GPU_SPACE_TIME_DIGGLE_SUMMARY
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test';
import {SPLANCS_FIXTURE} from './space-time-diggle-splancs-fixture';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const edgeCorrection of ['none', 'isotropic'] as const)
  for (const excluded of [false, true]) {
    it(`Diggle ${edgeCorrection}, excluded=${excluded} compares every output on whole-day times`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scene = SPLANCS_FIXTURE.integerTimes;
      const entry = scene.cases[1];
      const rowCount = scene.positions.length / 2;
      const props: SpaceTimeDiggleTestOperationProps = {
        positions: 'positions',
        times: 'times',
        mask: 'mask',
        parameters: 'parameters',
        randomParameters: 'random',
        rowCount,
        gridSize: [4, 4],
        distanceClassCount: scene.distanceClassCount,
        timeClassCount: entry.timeClassCount,
        permutationCount: 11,
        edgeCorrection,
        output: {
          d: 'd',
          statistics: 'statistics',
          summary: 'summary',
          permutationMean: 'mean',
          permutationStandardDeviation: 'deviation',
          k: 'k',
          spatialK: 'spatial',
          temporalK: 'temporal',
          status: 'status'
        }
      };
      const inputs = {
        positions: {format: 'float32x2' as const, values: Float32Array.from(scene.positions)},
        times: {format: 'float32' as const, values: Float32Array.from(scene.labellingTimes[0])},
        mask: {
          format: 'uint32' as const,
          values: Uint32Array.from({length: rowCount}, (_, row) =>
            excluded ? 0 : Number(row % 7 !== 0)
          )
        },
        parameters: {
          format: 'float32' as const,
          values: getGPUSpaceTimeKFunctionParameterValues({
            bounds: SPLANCS_FIXTURE.bounds,
            timeWindow: scene.timeLimits,
            maximumDistance: scene.maximumDistance,
            maximumTimeLag: entry.maximumTimeLag
          })
        },
        random: {
          format: 'uint32' as const,
          values: getGPUSpaceTimeDiggleRandomParameterValues(2 ** 40 + 21)
        }
      };
      const outputs = [
        'd',
        'statistics',
        'summary',
        'mean',
        'deviation',
        'k',
        'spatial',
        'temporal',
        'status'
      ];
      const gpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new GPUSpaceTimeDiggleTestOperation(props)]
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        outputs,
        operations: [new CPUSpaceTimeDiggleTestOperation(props)]
      });
      const scale = Math.max(...cpu['k'].filter(Number.isFinite).map(Math.abs), 1);
      for (const port of outputs)
        for (let index = 0; index < cpu[port].length; index++) {
          const expected = cpu[port][index];
          const actual = gpu[port][index];
          if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
          else if (
            port === 'status' ||
            (port === 'summary' &&
              new Set<number>([
                GPU_SPACE_TIME_DIGGLE_SUMMARY.greaterCount,
                GPU_SPACE_TIME_DIGGLE_SUMMARY.lesserCount,
                GPU_SPACE_TIME_DIGGLE_SUMMARY.usedClasses,
                GPU_SPACE_TIME_DIGGLE_SUMMARY.includedCount
              ]).has(index))
          )
            expect(actual === expected).toBe(true);
          else if (port === 'summary' && index === GPU_SPACE_TIME_DIGGLE_SUMMARY.zScore)
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-3);
          // Positive quantities use relative tolerance. Signed residual cancellation is scaled by K.
          else if (
            ['k', 'spatial', 'temporal', 'deviation'].includes(port) ||
            (port === 'summary' && [2, 5, 6, 7].includes(index))
          )
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-4 * Math.abs(expected));
          else
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
              2e-4 *
                scale *
                (port === 'statistics' || port === 'summary'
                  ? props.distanceClassCount * props.timeClassCount
                  : 1)
            );
        }
    });
  }
