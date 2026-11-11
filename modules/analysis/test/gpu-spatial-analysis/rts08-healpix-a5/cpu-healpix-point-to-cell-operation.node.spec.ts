// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeHealpixPointToCellOnCPU,
  CPUHealpixPointToCellOperation
} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';
import {
  GPUHealpixPointToCellOperation,
  type HealpixPointToCellOperationProps
} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {getHealpixNestedCenter} from './rts08-healpix-reference';

it('inverts the independent pixel-to-angle algorithm at every cell centre through order 4', () => {
  for (let order = 0; order <= 4; order++) {
    const count = 12 * 4 ** order;
    const positions = new Float32Array(count * 2);
    for (let pixel = 0; pixel < count; pixel++) {
      const center = getHealpixNestedCenter(BigInt(pixel), order);
      positions[pixel * 2] = center.longitude;
      positions[pixel * 2 + 1] = center.latitude;
    }
    for (const scheme of ['nested', 'uniq'] as const) {
      const result = computeHealpixPointToCellOnCPU(positions, {order, scheme});
      for (let pixel = 0; pixel < count; pixel++) {
        const key = BigInt(result.cells[pixel * 2]) | (BigInt(result.cells[pixel * 2 + 1]) << 32n);
        expect(key).toBe(BigInt(pixel) + (scheme === 'uniq' ? 1n << BigInt(2 * order + 2) : 0n));
        expect(result.validity[pixel]).toBe(1);
      }
    }
  }
});

it('preserves all 60 path bits at order 29 and handles masks, clamping, wrapping and invalid rows', () => {
  const result = computeHealpixPointToCellOnCPU(
    new Float32Array([0, 90, 0, -90, 360, 100, 0, 0, NaN, 0, 0, Infinity]),
    {order: 29, mask: new Uint32Array([1, 1, 1, 0, 1, 1])}
  );
  const getKey = (row: number) =>
    BigInt(result.cells[row * 2]) | (BigInt(result.cells[row * 2 + 1]) << 32n);
  // North face 0: both local coordinates nside-1; south face 8: both zero.
  expect(getKey(0)).toBe((1n << 58n) - 1n);
  expect(getKey(1)).toBe(8n << 58n);
  expect(getKey(2)).toBe(getKey(0));
  expect(Array.from(result.validity)).toEqual([1, 1, 1, 0, 0, 0]);
  expect(Array.from(result.cells.subarray(6))).toEqual([0, 0, 0, 0, 0, 0]);
  expect(() => computeHealpixPointToCellOnCPU(new Float32Array(3), {order: 0})).toThrow(
    /positions/
  );
  expect(() =>
    computeHealpixPointToCellOnCPU(new Float32Array(2), {order: 0, mask: new Uint32Array(2)})
  ).toThrow(/mask/);
});

it('validates both operation output shapes at construction and CPU order/context before execution', () => {
  const props: HealpixPointToCellOperationProps = {
    id: 'healpix-test',
    positions: 'positions',
    rowCount: 1,
    order: 0,
    output: {cells: 'cells', validity: 'validity'}
  };
  for (const Operation of [GPUHealpixPointToCellOperation, CPUHealpixPointToCellOperation]) {
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/^healpix-test .*length/);
    expect(() => new Operation({...props, output: {cells: 'positions'}})).toThrow(/^healpix-test /);
  }
  expect(() => new CPUHealpixPointToCellOperation({...props, order: 30})).toThrow(
    /^healpix-test order/
  );
  expect(
    () =>
      new CPUHealpixPointToCellOperation({
        ...props,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      })
  ).toThrow(/^healpix-test /);
});
