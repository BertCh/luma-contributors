// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterReconstructionOperation,
  type RasterReconstructionOperationProps
} from '../../../src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation';
import {CPURasterReconstructionOperation} from '../../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('reconstruction operations honor native uint32 bands, calibration, nodata and validity barriers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RasterReconstructionOperationProps = {
    grid: {width: 5, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    marker: {
      id: 'marker-band',
      format: 'uint32',
      storage: {kind: 'buffer', values: 'marker'},
      validity: 'marker-validity',
      scale: 2,
      noDataValue: 99
    },
    mask: {
      id: 'mask-band',
      format: 'uint32',
      storage: {kind: 'buffer', values: 'constraint'},
      scale: 2
    },
    maximumIterations: 16,
    output: {
      values: 'result',
      validity: 'validity',
      status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
    }
  };
  for (const Operation of [GPURasterReconstructionOperation, CPURasterReconstructionOperation]) {
    const operation = new Operation(props);
    const result = await runOperationGraph(device, {
      inputs: {
        marker: {format: 'uint32', values: Uint32Array.of(2, 0, 0, 99, 0)},
        'marker-validity': {format: 'uint32', values: Uint32Array.of(1, 1, 1, 1, 0)},
        constraint: {format: 'uint32', values: Uint32Array.of(3, 3, 3, 3, 3)}
      },
      operations: [operation],
      outputs: ['result', 'validity', 'converged', 'rounds', 'limit']
    });
    expect(Array.from(result['result'].slice(0, 3))).toEqual([4, 4, 4]);
    expect(result['result'][3]).toBeNaN();
    expect(result['result'][4]).toBeNaN();
    expect(Array.from(result['validity'])).toEqual([1, 1, 1, 0, 0]);
    expect(Array.from(result['converged'])).toEqual([1]);
    expect(Array.from(result['rounds'])).toEqual([3]);
    expect(Array.from(result['limit'])).toEqual([0]);
  }
});
