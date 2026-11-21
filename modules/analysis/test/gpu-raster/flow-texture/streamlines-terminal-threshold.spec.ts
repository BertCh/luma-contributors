// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('streamlines keep a supported below-threshold terminal point and take no next step', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      velocities: {format: 'float32x2', values: Float32Array.of(1, 0, 1, 0, 0, 0, 0, 0)},
      parameters: {
        format: 'float32',
        values: getGPUStreamlinesParameterValues({
          fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
          gridPlacement: {origin: [0, 0], cellSize: [1, 1]},
          stepLength: 1.375,
          minimumSpeed: 0.75
        })
      },
      words: {
        format: 'uint32',
        values: getGPUStreamlinesUint32ParameterValues({seed: 6, minimumPoints: 2})
      }
    },
    operations: [
      new GPUStreamlinesOperation({
        velocities: 'velocities',
        parameters: 'parameters',
        uint32Parameters: 'words',
        fieldWidth: 4,
        fieldHeight: 1,
        gridWidth: 4,
        gridHeight: 1,
        seedColumns: 1,
        seedRows: 1,
        stepsPerDirection: 2,
        maximumIterations: 32,
        lineCapacity: 1,
        pointCapacity: 5,
        output: {
          lines: {
            ids: 'ids',
            status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'}
          },
          lineOffsets: 'offsets',
          positions: 'positions',
          positionCount: 'positionCount'
        }
      })
    ],
    outputs: ['count', 'positionCount', 'positions']
  });
  expect(Array.from(rows['count'])).toEqual([1]);
  expect(Array.from(rows['positionCount'])).toEqual([2]);
  const positions = rows['positions'];
  // x0=.47235536575317383; supported terminal x=x0+1.375, whose speed is
  // 2.5-x=.6526446342468262 < .75. Existing semantics keep this point, then stop.
  expect(Number.isFinite(positions[1])).toBe(true);
  expect(Array.from(positions.subarray(0, 4))).toEqual([
    0.47235536575317383, 0.45453524589538574, 1.8473553657531738, 0.45453524589538574
  ]);
});
