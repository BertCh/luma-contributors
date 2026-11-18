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

for (const layout of ['exact', 'padded', 'offset'] as const) {
  it(`F8 both reconstruction operations calibrate the ${layout} logical band prefix`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const values =
      layout === 'exact'
        ? Float32Array.of(1)
        : layout === 'padded'
          ? Float32Array.of(1, 99)
          : Float32Array.of(99, 1, 77).subarray(1);
    const flags =
      layout === 'exact'
        ? Uint32Array.of(1)
        : layout === 'padded'
          ? Uint32Array.of(1, 0)
          : Uint32Array.of(0, 1, 0).subarray(1);
    const props: RasterReconstructionOperationProps = {
      grid: {width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
      marker: {
        id: 'marker-band',
        format: 'float32',
        storage: {kind: 'buffer', values: 'marker'},
        validity: 'marker-validity',
        noDataValue: 99
      },
      mask: {
        id: 'mask-band',
        format: 'float32',
        storage: {kind: 'buffer', values: 'mask'},
        validity: 'mask-validity',
        noDataValue: 99
      },
      maximumIterations: 1,
      output: {
        values: 'result',
        validity: 'validity',
        status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
      }
    };
    for (const Operation of [GPURasterReconstructionOperation, CPURasterReconstructionOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          marker: {format: 'float32', values},
          mask: {format: 'float32', values},
          'marker-validity': {format: 'uint32', values: flags},
          'mask-validity': {format: 'uint32', values: flags}
        },
        operations: [new Operation(props)],
        outputs: ['result', 'validity', 'converged', 'rounds', 'limit']
      });
      for (const [port, expected] of Object.entries({
        result: 1,
        validity: 1,
        converged: 1,
        rounds: 1,
        limit: 0
      }))
        expect(Array.from(result[port])).toEqual([expected]);
    }
  });
}

it('F8 reconstruction operation parity rejects undersized band values and validity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RasterReconstructionOperationProps = {
    grid: {width: 2, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
    marker: {
      id: 'marker-band',
      format: 'float32',
      storage: {kind: 'buffer', values: 'marker'},
      validity: 'marker-validity'
    },
    mask: {
      id: 'mask-band',
      format: 'float32',
      storage: {kind: 'buffer', values: 'mask'},
      validity: 'mask-validity'
    },
    maximumIterations: 1,
    output: {values: 'result', validity: 'validity', status: {converged: 'converged'}}
  };
  for (const shortPort of ['marker', 'mask', 'marker-validity', 'mask-validity']) {
    for (const Operation of [GPURasterReconstructionOperation, CPURasterReconstructionOperation]) {
      await expect(
        runOperationGraph(device, {
          inputs: {
            marker: {
              format: 'float32',
              values: new Float32Array(shortPort === 'marker' ? 1 : 2).fill(1)
            },
            mask: {
              format: 'float32',
              values: new Float32Array(shortPort === 'mask' ? 1 : 2).fill(1)
            },
            'marker-validity': {
              format: 'uint32',
              values: new Uint32Array(shortPort === 'marker-validity' ? 1 : 2).fill(1)
            },
            'mask-validity': {
              format: 'uint32',
              values: new Uint32Array(shortPort === 'mask-validity' ? 1 : 2).fill(1)
            }
          },
          operations: [new Operation(props)],
          outputs: ['result', 'validity', 'converged']
        })
      ).rejects.toThrow(/marker|mask/);
    }
  }
});
