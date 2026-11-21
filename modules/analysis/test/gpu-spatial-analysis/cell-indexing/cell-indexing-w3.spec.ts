// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUPointToCellOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation';
import {CPUPointToCellOperation} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {GPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {CPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {WRAPPED_CELL_KEYS} from './cell-wrap-fixtures';

for (const family of ['quadbin', 'quadkey', 'geohash', 'h3', 's2'] as const) {
  it(`DGN-13 named GPU ${family} wraps both latitudes to independent literal keys`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = await runOperationGraph(device, {
      inputs: {
        positions: {
          format: 'float32x2',
          values: Float32Array.of(210, 0, -150, 0, 210, 10, -150, 10)
        }
      },
      operations: [
        new GPUPointToCellOperation({
          family,
          resolution: 5,
          rowCount: 4,
          longitudePolicy: 'wrap',
          precision: family === 'h3' ? 'double-single' : 'float32',
          positions: 'positions',
          output: {cells: 'cells', validity: 'validity'}
        })
      ],
      outputs: ['cells', 'validity']
    });
    expect(rows['cells']).toHaveLength(8);
    const keys = Array.from(
      {length: 4},
      (_, row) => BigInt(rows['cells'][row * 2]) | (BigInt(rows['cells'][row * 2 + 1]) << 32n)
    );
    expect(keys).toEqual([
      WRAPPED_CELL_KEYS[family][0],
      WRAPPED_CELL_KEYS[family][0],
      WRAPPED_CELL_KEYS[family][1],
      WRAPPED_CELL_KEYS[family][1]
    ]);
    expect(Array.from(rows['validity'])).toEqual([1, 1, 1, 1]);
  });
}

it('FRM-12 empty CPU and GPU named cell stages return exact empty arrays', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    positions: 'positions',
    rowCount: 0,
    output: {cells: 'cells', validity: 'validity'}
  };
  for (const operation of [
    new GPUPointToCellOperation({...props, family: 'quadbin', resolution: 5}),
    new CPUPointToCellOperation({...props, family: 'quadbin', resolution: 5}),
    new GPUHealpixPointToCellOperation({...props, order: 1}),
    new CPUHealpixPointToCellOperation({...props, order: 1})
  ]) {
    const rows = await runOperationGraph(device, {
      inputs: {positions: {format: 'float32x2', values: new Float32Array()}},
      operations: [operation],
      outputs: ['cells', 'validity']
    });
    expect(rows['cells']).toEqual(new Uint32Array());
    expect(rows['validity']).toEqual(new Uint32Array());
  }
});

it('S3 wrapped GPU quadkey keeps the negative subnormal on the west side of zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {positions: {format: 'float32x2', values: Float32Array.of(-(2 ** -149), 0, 0, 0)}},
    operations: [
      new GPUPointToCellOperation({
        family: 'quadkey',
        resolution: 5,
        longitudePolicy: 'wrap',
        rowCount: 2,
        positions: 'positions',
        output: {cells: 'cells', validity: 'validity'}
      })
    ],
    outputs: ['cells', 'validity']
  });
  // x=15,y=16 => Morton 597; x=16,y=16 => Morton 768, with zoom 5 in the high word.
  expect(rows['cells']).toEqual(Uint32Array.of(597, 0x14000000, 768, 0x14000000));
  expect(rows['validity']).toEqual(Uint32Array.of(1, 1));
});

it('HEALPix GPU wraps reciprocal-division residuals to the same west-of-meridian key', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: Float32Array.of(1799.9998779296875, 0, 359.9998779296875, 0)
      }
    },
    operations: [
      new GPUHealpixPointToCellOperation({
        order: 11,
        rowCount: 2,
        positions: 'positions',
        output: {cells: 'cells', validity: 'validity'}
      })
    ],
    outputs: ['cells', 'validity']
  });
  // Face 4, x=1023,y=1024; 4*4^11 + Morton(x,y) = 19223893 for both angles.
  expect(rows['cells']).toEqual(Uint32Array.of(19223893, 0, 19223893, 0));
  expect(rows['validity']).toEqual(Uint32Array.of(1, 1));
});
