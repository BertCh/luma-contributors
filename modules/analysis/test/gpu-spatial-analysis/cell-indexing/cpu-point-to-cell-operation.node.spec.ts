// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getS2BoundaryFlat, getS2Level} from '@math.gl/dggs';
import {getPentagons, cellToLatLng, cellToBoundary, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {
  computePointToCellOnCPU,
  CPUPointToCellOperation
} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {
  GPUPointToCellOperation,
  type PointToCellOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation';
import {GPU_CELL_INDEX_FAMILIES} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-index-families';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import mercatorFixtures from './point-to-cell-operation-fixture.json';

const PROPS: PointToCellOperationProps = {
  id: 'point-test',
  family: 's2',
  resolution: 5,
  rowCount: 2,
  positions: 'positions',
  mask: 'mask',
  output: {cells: 'cells', validity: 'validity'}
};

function getKey(words: Uint32Array, row: number): bigint {
  return BigInt(words[row * 2]) | (BigInt(words[row * 2 + 1]) << 32n);
}

it('matches compiled H3 on float32 positions at every resolution, including pentagon neighborhoods', () => {
  for (let resolution = 0; resolution <= 15; resolution++) {
    const positions: number[] = [0, 90, 0, -90, -180, 0, 180, 0, 0, 0, 179.75, -47.25];
    for (let row = 0; row < 200; row++) {
      // A deterministic equal-area lattice, with unrelated longitude and latitude strides.
      positions.push(
        ((row * 137.50776405) % 360) - 180,
        (Math.asin((2 * (row + 0.5)) / 200 - 1) * 180) / Math.PI
      );
    }
    for (const pentagon of getPentagons(resolution)) {
      const [latitude, longitude] = cellToLatLng(pentagon);
      for (const offset of [-0.0001, 0, 0.0001])
        positions.push(longitude + offset, latitude - offset);
    }
    const boundaryStartRow = positions.length / 2;
    for (const [longitude, latitude] of [
      [-122.4194, 37.7749],
      [10.40744, 57.64911],
      [179.9, -32.75]
    ]) {
      const boundary = cellToBoundary(latLngToCell(latitude, longitude, resolution));
      for (const [boundaryLatitude, boundaryLongitude] of boundary) {
        for (const offset of [-1e-7, 0, 1e-7])
          positions.push(boundaryLongitude + offset, boundaryLatitude - offset);
      }
    }
    const packed = Float32Array.from(positions);
    expect(packed.length / 2).toBeGreaterThan(boundaryStartRow);
    for (const precision of ['float32', 'double-single'] as const) {
      const result = computePointToCellOnCPU(packed, {family: 'h3', resolution, precision});
      for (let row = 0; row < packed.length / 2; row++) {
        expect(getKey(result.cells, row), `resolution ${resolution}, row ${row}`).toBe(
          BigInt(`0x${latLngToCell(packed[row * 2 + 1], packed[row * 2], resolution)}`)
        );
        expect(result.validity[row]).toBe(1);
      }
    }
  }
});

it('preserves the integer-defined resolution-26 row and packed keys', () => {
  const positions = Float32Array.of(-8.913032531738281, 53.91400146484375);
  // Worked integer result: column 31892922, row 21574670. Interleaving their 26 bits
  // gives 0x374964e3145ec; Quadbin at level 26 has no footer bits.
  for (const [family, expected] of [
    ['quadbin', 0x49a374964e3145ecn],
    ['quadkey', 0x680374964e3145ecn]
  ] as const) {
    const result = computePointToCellOnCPU(positions, {family, resolution: 26});
    expect(getKey(result.cells, 0)).toBe(expected);
    expect(Array.from(result.validity)).toEqual([1]);
  }
});

it('checks the previously discarded H3 resolution-zero boundary row', () => {
  // Resolution zero, mode one, base cell 40, all fifteen unused digits equal seven.
  const expected = (1n << 59n) | (40n << 45n) | ((1n << 45n) - 1n);
  expect(expected).toBe(0x8051fffffffffffn);
  for (const precision of ['float32', 'double-single'] as const) {
    const result = computePointToCellOnCPU(
      Float32Array.of(-121.33663177490234, 28.653018951416016),
      {family: 'h3', resolution: 0, precision}
    );
    expect(getKey(result.cells, 0)).toBe(expected);
    expect(Array.from(result.validity)).toEqual([1]);
  }
});

it('agrees with independent PROJ at coarse levels and stays within its fine-level row error bound', () => {
  for (const fixture of mercatorFixtures)
    for (const family of ['quadbin', 'quadkey'] as const) {
      if (
        (family === 'quadbin' && fixture.level > 26) ||
        (family === 'quadkey' && fixture.level === 0)
      )
        continue;
      const result = computePointToCellOnCPU(Float32Array.of(fixture.longitude, fixture.latitude), {
        family,
        resolution: fixture.level
      });
      const key = getKey(result.cells, 0);
      const path =
        family === 'quadkey'
          ? key & ((1n << 58n) - 1n)
          : (key & ((1n << 52n) - 1n)) >> BigInt(52 - 2 * fixture.level);
      let column = 0;
      let tileRow = 0;
      for (let bit = 0; bit < fixture.level; bit++) {
        column += Number((path >> BigInt(2 * bit)) & 1n) * 2 ** bit;
        tileRow += Number((path >> BigInt(2 * bit + 1)) & 1n) * 2 ** bit;
      }
      expect(column).toBe(fixture.column);
      // PROJ evaluates continuous Mercator, while the contributor defines rows with Q0.32
      // polynomials and logarithms. At fine levels its documented error can cross one tile.
      if (fixture.level <= 15) expect(tileRow).toBe(fixture.row);
      else expect(Math.abs(tileRow - fixture.row)).toBeLessThanOrEqual(1);
    }
});

it('S2 leaf keys contain the point according to an independent inverse Hilbert decoder', () => {
  // math.gl decodes Hilbert IDs to spherical corners. Four oriented great-circle halfspaces
  // check containment, independently of the forward projection/quantization in the twin.
  const getUnit = (longitude: number, latitude: number) => {
    const longitudeAngle = (longitude * Math.PI) / 180;
    const latitudeAngle = (latitude * Math.PI) / 180;
    return [
      Math.cos(latitudeAngle) * Math.cos(longitudeAngle),
      Math.cos(latitudeAngle) * Math.sin(longitudeAngle),
      Math.sin(latitudeAngle)
    ];
  };
  const getCross = (first: number[], second: number[]) => [
    first[1] * second[2] - first[2] * second[1],
    first[2] * second[0] - first[0] * second[2],
    first[0] * second[1] - first[1] * second[0]
  ];
  const getDot = (first: number[], second: number[]) =>
    first.reduce((sum, value, axis) => sum + value * second[axis], 0);
  const positions = Float32Array.from(
    Array.from({length: 360}, (_, row) => [
      ((row * 137.50776405) % 360) - 180,
      (Math.asin((2 * (row + 0.5)) / 360 - 1) * 180) / Math.PI
    ]).flat()
  );
  for (const level of [1, 10, 26, 30]) {
    const result = computePointToCellOnCPU(positions, {family: 's2', resolution: level});
    for (let row = 0; row < positions.length / 2; row++) {
      const key = getKey(result.cells, row);
      expect(getS2Level(key)).toBe(level);
      const boundary = getS2BoundaryFlat(key);
      const edgeSampleCount = (boundary.length - 2) / 8;
      const corners = Array.from({length: 4}, (_, corner) =>
        getUnit(
          boundary[(corner + 1) * edgeSampleCount * 2 - 2],
          boundary[(corner + 1) * edgeSampleCount * 2 - 1]
        )
      );
      const center = corners[0].map((_, axis) =>
        corners.reduce((sum, corner) => sum + corner[axis], 0)
      );
      const point = getUnit(positions[row * 2], positions[row * 2 + 1]);
      for (let edge = 0; edge < 4; edge++) {
        // Cross products of neighboring leaf vertices lose ~1e-16 absolute precision. Dividing
        // by their ~1e-9 separation would amplify it, so use a stable edge difference first.
        const start = corners[edge];
        const difference = corners[(edge + 1) % 4].map((value, axis) => value - start[axis]);
        const stableNormal = getCross(start, difference);
        const centerOffset = center.map((value, axis) => value - 4 * start[axis]);
        const pointOffset = point.map((value, axis) => value - start[axis]);
        expect(
          (Math.sign(getDot(stableNormal, centerOffset)) * getDot(stableNormal, pointOffset)) /
            Math.hypot(...stableNormal)
        ).toBeGreaterThanOrEqual(-2e-15);
      }
    }
  }
});

it('matches hand-computed Mercator quadrants and canonical key packing', () => {
  const positions = Float32Array.of(-90, 45, 90, 45, -90, -45, 90, -45);
  const quadkeys = computePointToCellOnCPU(positions, {family: 'quadkey', resolution: 1}).cells;
  const quadbins = computePointToCellOnCPU(positions, {family: 'quadbin', resolution: 1}).cells;
  for (let row = 0; row < 4; row++) {
    // NW, NE, SW, SE have Morton digits 0, 1, 2, 3. Quadbin has 50 footer one bits.
    expect(getKey(quadkeys, row)).toBe((1n << 58n) | BigInt(row));
    expect(getKey(quadbins, row)).toBe(
      0x4810000000000000n | (BigInt(row) << 50n) | ((1n << 50n) - 1n)
    );
  }
  for (const family of ['quadbin', 'quadkey'] as const) {
    const result = computePointToCellOnCPU(Float32Array.of(-180, 0, 180, 0, -200, 100, 200, 90), {
      family,
      resolution: 1
    });
    expect(getKey(result.cells, 0)).toBe(getKey(result.cells, 1));
    expect(getKey(result.cells, 2)).toBe(getKey(result.cells, 3));
  }
});

it('matches known geohash u4pru and exact dyadic boundary choices at length twelve', () => {
  const result = computePointToCellOnCPU(Float32Array.of(10.40744, 57.64911), {
    family: 'geohash',
    resolution: 5
  });
  // Published geohash example u4pruydqqvj, truncated to five characters. Alphabet codes:
  const codes = [26, 4, 21, 23, 26];
  const payload = codes.reduce((key, code) => (key << 5n) | BigInt(code), 0n);
  expect(getKey(result.cells, 0)).toBe((5n << 60n) | payload);
  const extremes = computePointToCellOnCPU(Float32Array.of(-180, -90, 180, 90, 0, 0, -200, -100), {
    family: 'geohash',
    resolution: 12
  });
  expect(getKey(extremes.cells, 0)).toBe(12n << 60n);
  expect(getKey(extremes.cells, 1)).toBe((13n << 60n) - 1n);
  expect(getKey(extremes.cells, 2)).toBe((12n << 60n) | (3n << 58n));
  expect(getKey(extremes.cells, 3)).toBe(getKey(extremes.cells, 0));
});

it('uses S2 face IDs and preserves poles, antimeridian equivalence, masks and invalid rows', () => {
  const axes = computePointToCellOnCPU(
    Float32Array.of(0, 0, 90, 0, 0, 90, 180, 0, -90, 0, 0, -90),
    {family: 's2', resolution: 0}
  );
  for (let face = 0; face < 6; face++)
    expect(getKey(axes.cells, face)).toBe(BigInt(2 * face + 1) << 60n);
  for (const family of GPU_CELL_INDEX_FAMILIES) {
    const resolution = family === 's2' ? 30 : 3;
    const result = computePointToCellOnCPU(
      Float32Array.of(12, 23, 50, 20, NaN, 0, 0, Infinity),
      {family, resolution},
      Uint32Array.of(7, 0, 1, 1)
    );
    expect(Array.from(result.validity)).toEqual([1, 0, 0, 0]);
    expect(Array.from(result.cells.subarray(2))).toEqual([0, 0, 0, 0, 0, 0]);
    expect(computePointToCellOnCPU(new Float32Array(), {family, resolution}).cells.length).toBe(0);
  }
  const result = computePointToCellOnCPU(
    Float32Array.of(-180, 12, 180, 12, 0, 90, 127, 90, 0, 100),
    {family: 's2', resolution: 30}
  );
  expect(getKey(result.cells, 0)).toBe(getKey(result.cells, 1));
  expect(getKey(result.cells, 2)).toBe(getKey(result.cells, 3));
  expect(getKey(result.cells, 2)).toBe(getKey(result.cells, 4));
});

it('validates arrays and construction of both operations, including deferred contributor construction', () => {
  expect(() => computePointToCellOnCPU(new Float32Array(3), PROPS)).toThrow(/positions/);
  expect(() => computePointToCellOnCPU(new Float32Array(4), PROPS, new Uint32Array(3))).toThrow(
    /mask/
  );
  const shapes = {
    positions: {format: 'float32x2' as const, length: 2},
    mask: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUPointToCellOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUPointToCellOperation, CPUPointToCellOperation]) {
    expect(() => new Operation({...PROPS, rowCount: -1})).toThrow(/point-test/);
    expect(() => new Operation({...PROPS, output: {cells: 'positions'}})).toThrow(/point-test/);
    expect(() => new Operation({...PROPS, output: {cells: 'same', validity: 'same'}})).toThrow(
      /point-test/
    );
  }
  for (const resolution of [-1, 31, 0.5, NaN]) {
    expect(() => new CPUPointToCellOperation({...PROPS, resolution})).toThrow(
      /point-test.*resolution/
    );
    expect(() =>
      constructNetworkOperationContributor(
        new GPUPointToCellOperation({...PROPS, resolution}),
        shapes
      )
    ).toThrow(/point-test.*resolution/);
  }
  const spatialContext = {coordinateSpace: 'planar' as const, metric: 'native' as const};
  expect(() => new CPUPointToCellOperation({...PROPS, spatialContext})).toThrow(/point-test/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPUPointToCellOperation({...PROPS, spatialContext}),
      shapes
    )
  ).toThrow(/point-test/);
  expect(() =>
    constructNetworkOperationContributor(new GPUPointToCellOperation(PROPS), {
      ...shapes,
      mask: {format: 'uint32', length: 1}
    })
  ).toThrow(/point-test.*mask/);
});
