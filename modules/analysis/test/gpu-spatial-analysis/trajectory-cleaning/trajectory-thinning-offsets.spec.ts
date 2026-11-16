// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {getGPUTrajectoryThinningParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('thins independent five-row tracks and publishes their closed-form offsets in parallel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const trackCount = 65537;
  const rowCount = trackCount * 5;
  const expectedRows = Array.from({length: trackCount}, (_, track) => [
    track * 5,
    track * 5 + 2,
    track * 5 + 4
  ]).flat();
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array(rowCount * 2)},
      timestamps: {
        format: 'float32',
        values: Float32Array.from({length: rowCount}, (_, row) => row % 5)
      },
      trackOffsets: {
        format: 'uint32',
        values: Uint32Array.from({length: trackCount + 1}, (_, track) => track * 5)
      },
      parameters: {
        format: 'float32',
        values: getGPUTrajectoryThinningParameterValues({tolerance: 2})
      }
    },
    operations: [
      new GPUTrajectoryThinningOperation({
        mode: 'minimum-time',
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        parameters: 'parameters',
        rowCount,
        trackCount,
        capacity: expectedRows.length,
        output: {
          rowIds: 'rows',
          trackOffsets: 'offsets',
          status: {count: 'count', overflow: 'overflow'}
        }
      })
    ],
    outputs: ['rows', 'offsets', 'count', 'overflow']
  });
  expect(Array.from(result['rows'])).toEqual(expectedRows);
  expect(Array.from(result['offsets'])).toEqual(
    Array.from({length: trackCount + 1}, (_, track) => track * 3)
  );
  expect(Array.from(result['count'])).toEqual([expectedRows.length]);
  expect(Array.from(result['overflow'])).toEqual([0]);
}, 120000);
