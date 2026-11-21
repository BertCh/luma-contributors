// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {calibrateRasterOperationBand} from '../../../src/raster-analysis/raster-grid/raster-operation-band';
import {
  computeTerrainPeakProminenceOnCPU,
  CPUTerrainPeakProminenceOperation
} from '../../../src/gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation';
import {
  GPUTerrainPeakProminenceOperation,
  type TerrainPeakProminenceOperationProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence-operation';

const GRID = {
  width: 5,
  height: 1,
  pixelInterpretation: 'area' as const,
  affine: [1, 0, 0, 0, -1, 0] as const
};
const NONE = 0xffffffff;

it('rejects native nodata outside the scalar format at construction and in the pure function', () => {
  const grid = {...GRID, width: 1};
  const props: TerrainPeakProminenceOperationProps = {
    id: 'native-nodata',
    grid,
    elevation: {
      id: 'elevation',
      format: 'uint32',
      storage: {kind: 'buffer', values: 'values'},
      noDataValue: -1
    },
    output: {prominence: 'prominence'}
  };
  // A uint32 sentinel is an integer in [0, 4294967295]; -1 cannot represent one.
  expect(() => new CPUTerrainPeakProminenceOperation(props)).toThrow(
    /^native-nodata.*noDataValue.*uint32/
  );
  expect(() =>
    computeTerrainPeakProminenceOnCPU(
      {id: 'elevation', format: 'uint32', values: new Uint32Array([10]), noDataValue: -1},
      grid
    )
  ).toThrow(/noDataValue.*uint32/);
  for (const [format, noDataValue] of [
    ['uint32', 4294967296],
    ['uint32', 0.5],
    ['uint32', NaN],
    ['sint32', -2147483649],
    ['sint32', 2147483648],
    ['sint32', 0.5],
    ['float32', Infinity],
    ['float32', 1e100]
  ] as const) {
    expect(
      () =>
        new CPUTerrainPeakProminenceOperation({
          ...props,
          elevation: {...props.elevation, format, noDataValue}
        })
    ).toThrow(/native-nodata.*noDataValue/);
  }
  for (const [format, noDataValue] of [
    ['uint32', 0],
    ['uint32', 4294967295],
    ['sint32', -2147483648],
    ['sint32', 2147483647],
    ['float32', NaN],
    ['float32', 0.1]
  ] as const) {
    expect(
      () =>
        new CPUTerrainPeakProminenceOperation({
          ...props,
          elevation: {...props.elevation, format, noDataValue}
        })
    ).not.toThrow();
  }
});

it('canonicalizes a negative-zero edge saddle to positive zero', () => {
  const result = computeTerrainPeakProminenceOnCPU(
    {id: 'elevation', format: 'float32', values: new Float32Array([-0, -0]), offset: -0},
    {...GRID, width: 2}
  );
  // Equal heights keep elder cell 0; cell 1 merges at the canonical +0 saddle.
  expect(Array.from(result.keySaddle)).toEqual([-Infinity, 0]);
  expect(Object.is(result.keySaddle[1], -0)).toBe(false);
  expect(Array.from(result.parentPeak)).toEqual([NONE, 0]);
  expect(result.prominence[0]).toBe(Infinity);
  expect(result.prominence[1] === 0).toBe(true);
});

it('matches hand-computed saddle and stable parent ties for nested peaks and a plateau', () => {
  const result = computeTerrainPeakProminenceOnCPU(
    {id: 'elevation', format: 'float32', values: new Float32Array([10, 2, 8, 3, 9])},
    GRID
  );
  expect(Array.from(result.prominence)).toEqual([Infinity, 0, 5, 0, 7]);
  expect(Array.from(result.keySaddle)).toEqual([-Infinity, 2, 3, 3, 2]);
  expect(Array.from(result.parentPeak)).toEqual([NONE, 0, 4, 2, 0]);
  const plateau = computeTerrainPeakProminenceOnCPU(
    {id: 'elevation', format: 'float32', values: new Float32Array([5, 5, 5, 5, 5])},
    GRID
  );
  expect(Array.from(plateau.prominence)).toEqual([Infinity, 0, 0, 0, 0]);
  expect(Array.from(plateau.parentPeak)).toEqual([NONE, 0, 0, 0, 0]);
});

it('agrees with an independent all-pairs widest-path oracle on a tiny 8-connected grid', () => {
  const grid = {...GRID, width: 3, height: 3};
  const values = new Float32Array([9, 1, 8, 2, NaN, 3, 7, 4, 6]);
  const count = values.length;
  // Floyd-Warshall in the max/min semiring, rather than sorted edges and union-find.
  const capacity = Array.from({length: count}, () => Array<number>(count).fill(-Infinity));
  for (let first = 0; first < count; first++) {
    if (!Number.isFinite(values[first])) continue;
    capacity[first][first] = values[first];
    for (let second = 0; second < count; second++) {
      if (
        Number.isFinite(values[second]) &&
        Math.abs((first % 3) - (second % 3)) <= 1 &&
        Math.abs(Math.floor(first / 3) - Math.floor(second / 3)) <= 1
      )
        capacity[first][second] = Math.min(values[first], values[second]);
    }
  }
  for (let middle = 0; middle < count; middle++)
    for (let first = 0; first < count; first++)
      for (let second = 0; second < count; second++)
        capacity[first][second] = Math.max(
          capacity[first][second],
          Math.min(capacity[first][middle], capacity[middle][second])
        );
  const result = computeTerrainPeakProminenceOnCPU(
    {id: 'elevation', format: 'float32', values},
    grid
  );
  for (let cell = 0; cell < count; cell++) {
    if (!Number.isFinite(values[cell])) {
      expect(result.prominence[cell]).toBeNaN();
      expect(result.keySaddle[cell]).toBeNaN();
      expect(result.parentPeak[cell]).toBe(NONE);
      continue;
    }
    let saddle = -Infinity;
    for (let elder = 0; elder < count; elder++)
      if (values[elder] > values[cell] || (values[elder] === values[cell] && elder < cell))
        saddle = Math.max(saddle, capacity[cell][elder]);
    expect(result.keySaddle[cell]).toBe(saddle);
    expect(result.prominence[cell]).toBe(values[cell] - saddle);
  }
});

it('honors exact integer raw nodata, calibration, validity barriers and disconnected elders', () => {
  const result = computeTerrainPeakProminenceOnCPU(
    {
      id: 'elevation',
      format: 'sint32',
      values: new Int32Array([10, -999, 8, 3, 9]),
      noDataValue: -999,
      scale: 2,
      offset: 1,
      validity: new Uint32Array([1, 1, 1, 0, 1])
    },
    GRID
  );
  expect(Array.from(result.prominence)).toEqual([Infinity, NaN, Infinity, NaN, Infinity]);
  expect(Array.from(result.keySaddle)).toEqual([-Infinity, NaN, -Infinity, NaN, -Infinity]);
  expect(Array.from(result.parentPeak)).toEqual([NONE, NONE, NONE, NONE, NONE]);
  const roundedNoData = computeTerrainPeakProminenceOnCPU(
    {
      id: 'elevation',
      format: 'float32',
      values: new Float32Array([0.1, 2, 3, 4, 5]),
      noDataValue: 0.1
    },
    GRID
  );
  expect(roundedNoData.prominence[0]).toBeNaN();
  expect(roundedNoData.parentPeak[0]).toBe(NONE);
});

it('validates both declarations, band metadata and inconsistent arrays', () => {
  const props: TerrainPeakProminenceOperationProps = {
    id: 'prominence-validation',
    grid: GRID,
    elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: 'values'}},
    output: {prominence: 'prominence'}
  };
  for (const Operation of [CPUTerrainPeakProminenceOperation, GPUTerrainPeakProminenceOperation]) {
    expect(() => new Operation({...props, grid: {...GRID, width: 0}})).toThrow(
      /^prominence-validation/
    );
    expect(() => new Operation({...props, output: {prominence: 'values'}})).toThrow(
      /^prominence-validation/
    );
  }
  expect(
    () =>
      new CPUTerrainPeakProminenceOperation({...props, elevation: {...props.elevation, scale: NaN}})
  ).toThrow(/prominence-validation.*scale/);
  expect(() =>
    computeTerrainPeakProminenceOnCPU(
      {id: 'elevation', format: 'float32', values: new Float32Array(4)},
      GRID
    )
  ).toThrow(/values/);
  expect(() =>
    computeTerrainPeakProminenceOnCPU(
      {
        id: 'elevation',
        format: 'float32',
        values: new Float32Array(5),
        validity: new Uint32Array(4)
      },
      GRID
    )
  ).toThrow(/validity/);
});

it('TWN-08 retains integer calibration differences and the true elder in the pure and CPU operation forms', async () => {
  const grid = {...GRID, width: 2};
  const values = Int32Array.of(16777216, 16777217);
  const elevation = {
    id: 'elevation',
    format: 'sint32' as const,
    values,
    scale: 1,
    offset: -16777216
  };
  // Exact integer arithmetic: [2^24, 2^24+1] - 2^24 = [0,1]. Cell 1 is elder;
  // cell 0 joins at saddle 0 and has prominence 0. This is not a plateau tie.
  expect(Array.from(calibrateRasterOperationBand(elevation, 2))).toEqual([0, 1]);
  const pure = computeTerrainPeakProminenceOnCPU(elevation, grid);
  expect(Array.from(pure.parentPeak)).toEqual([1, NONE]);
  expect(Array.from(pure.prominence)).toEqual([0, Infinity]);
  const operation = new CPUTerrainPeakProminenceOperation({
    grid,
    elevation: {...elevation, storage: {kind: 'buffer', values: 'values'}},
    output: {prominence: 'prominence', parentPeak: 'parent'}
  });
  const rows = await operation.execute({
    inputs: {
      values: {
        format: 'sint32',
        length: 2,
        chunks: [{values, length: 2}],
        pack: () => values
      }
    }
  });
  expect(Array.from(rows['parent'] as Uint32Array)).toEqual([1, NONE]);
  expect(Array.from(rows['prominence'] as Float32Array)).toEqual([0, Infinity]);
});

it('S3 names elevation when the prominence CPU boundary rejects mismatched storage', () => {
  const malformed = {id: 'dem', format: 'sint32', values: Float32Array.of(1)};
  expect(() => computeTerrainPeakProminenceOnCPU(malformed as never, {...GRID, width: 1})).toThrow(
    /^computeTerrainPeakProminenceOnCPU.*elevation.values must be Int32Array/
  );
});
