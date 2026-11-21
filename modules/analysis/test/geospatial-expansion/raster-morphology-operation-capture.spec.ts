// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {GPURasterReconstructionOperation} from '../../src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation';
import {GPURasterFillHolesOperation} from '../../src/raster-analysis/raster-morphology/gpu-raster-fill-holes-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';

class InheritedStatus {
  get converged() {
    return 'done';
  }
  get iterationCount() {
    return 'rounds';
  }
  get iterationLimitReached() {
    return 'limited';
  }
}

it('W5-09 reconstruction writes inherited band and convergence destinations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const marker = new (class {
    get id() {
      return 'marker-band';
    }
    get format() {
      return 'float32' as const;
    }
    get storage() {
      return new (class {
        get kind() {
          return 'buffer' as const;
        }
        get values() {
          return 'marker';
        }
      })();
    }
    get validity() {
      return 'marker-validity';
    }
    get scale() {
      return 2;
    }
  })();
  const result = await runOperationGraph(device, {
    inputs: {
      marker: {format: 'float32', values: Float32Array.of(1, 0, 0, 0, 0)},
      mask: {format: 'float32', values: new Float32Array(5).fill(2)},
      'marker-validity': {format: 'uint32', values: Uint32Array.of(1, 1, 1, 1, 0)}
    },
    operations: [
      new GPURasterReconstructionOperation({
        grid: {width: 5, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        marker,
        mask: 'mask',
        maximumIterations: 1,
        output: {values: 'values', validity: 'validity', status: new InheritedStatus()}
      })
    ],
    outputs: ['values', 'validity', 'done', 'rounds', 'limited']
  });
  // Last marker row is invalid; the first four cells still exercise one bounded propagation round.
  expect(Array.from(result['values'])).toEqual([2, 2, 0, 0, NaN]);
  expect(Array.from(result['validity'])).toEqual([1, 1, 1, 1, 0]);
  expect(Array.from(result['done'])).toEqual([0]);
  expect(Array.from(result['rounds'])).toEqual([2]);
  expect(Array.from(result['limited'])).toEqual([1]);
});

it('W5-09 fill holes writes inherited convergence destinations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {input: {format: 'uint32', values: new Uint32Array(25)}},
    operations: [
      new GPURasterFillHolesOperation({
        grid: {width: 5, height: 5, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        input: 'input',
        maximumIterations: 1,
        output: {values: 'filled', status: new InheritedStatus()}
      })
    ],
    outputs: ['filled', 'done', 'rounds', 'limited']
  });
  // Boundary depth zero starts flooded; one round reaches depth one, leaving only center depth two.
  expect(Array.from(result['filled'])).toEqual([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  expect(Array.from(result['done'])).toEqual([0]);
  expect(Array.from(result['rounds'])).toEqual([2]);
  expect(Array.from(result['limited'])).toEqual([1]);
});

for (const format of ['uint32', 'sint32'] as const) {
  it(`B3 GPU reconstruction converts ${format} samples before calibration`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const ArrayType = format === 'uint32' ? Uint32Array : Int32Array;
    const result = await runOperationGraph(device, {
      inputs: {
        marker: {format, values: new ArrayType([33554433])},
        mask: {format: 'float32', values: Float32Array.of(0)}
      },
      operations: [
        new GPURasterReconstructionOperation({
          grid: {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          marker: {
            id: 'marker',
            format,
            offset: -33554432,
            storage: {kind: 'buffer', values: 'marker'}
          },
          mask: 'mask',
          output: {values: 'values', validity: 'validity', status: new InheritedStatus()}
        })
      ],
      outputs: ['values', 'validity', 'done', 'rounds', 'limited']
    });
    // Binary32 spacing at 2^25 is four: 33554433 converts to 33554432, then offset cancels it.
    expect(Array.from(result['values'])).toEqual([0]);
    expect(Array.from(result['validity'])).toEqual([1]);
    expect(Array.from(result['done'])).toEqual([1]);
    expect(Array.from(result['rounds'])).toEqual([1]);
    expect(Array.from(result['limited'])).toEqual([0]);
  });
}
