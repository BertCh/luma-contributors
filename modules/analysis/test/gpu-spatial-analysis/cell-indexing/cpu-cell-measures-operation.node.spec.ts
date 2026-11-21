// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellArea, cellToBoundary, getPentagons, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {
  computeCellMeasuresOnCPU,
  CPUCellMeasuresOperation
} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation';
import {
  GPUCellMeasuresOperation,
  type CellMeasuresOperationProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {H3_MEASURES_FIXTURES} from './h3-measures-fixtures';

const PROPS: CellMeasuresOperationProps = {
  id: 'measure-test',
  family: 'h3',
  cells: 'cells',
  rowCount: 2,
  maximumEdgeCount: 10,
  output: {
    areas: 'areas',
    perimeters: 'perimeters',
    edgeLengths: 'edgeLengths',
    edgeCounts: 'edgeCounts',
    pentagons: 'pentagons'
  }
};

function packKeys(keys: string[], highLow = false): Uint32Array {
  return Uint32Array.from(
    keys.flatMap(key => {
      const value = BigInt(`0x${key}`);
      const words = [Number(value & 0xffffffffn), Number(value >> 32n)];
      return highLow ? words.reverse() : words;
    })
  );
}

it('matches independently generated python H3 areas, perimeters, ordered edges and pentagons', () => {
  const result = computeCellMeasuresOnCPU(packKeys(H3_MEASURES_FIXTURES.map(row => row[0])), {
    family: 'h3',
    maximumEdgeCount: 10,
    units: 'kilometers'
  });
  H3_MEASURES_FIXTURES.forEach(([key, area, perimeter, count, , pentagon, edges], row) => {
    expect(result.edgeCounts[row], key).toBe(count);
    expect(result.pentagons[row], key).toBe(pentagon);
    expect(Math.abs(result.areas[row] / area - 1), `${key} area`).toBeLessThan(2e-5);
    expect(Math.abs(result.perimeters[row] / perimeter - 1), `${key} perimeter`).toBeLessThan(2e-5);
    edges.forEach((edge, index) =>
      expect(
        Math.abs(result.edgeLengths[row * 10 + index] / edge - 1),
        `${key} edge ${index}`
      ).toBeLessThan(2e-5)
    );
    expect(Array.from(result.edgeLengths.subarray(row * 10 + count, (row + 1) * 10))).toEqual(
      new Array(10 - count).fill(0)
    );
  });
});

it('decodes pentagons and their neighborhoods at all resolutions against compiled H3', () => {
  for (let resolution = 0; resolution <= 15; resolution++) {
    const keys = [...new Set(getPentagons(resolution).flatMap(key => gridDisk(key, 2)))];
    // Keep this compiled-H3 oracle off the longitude seam: H3's inverse trig loses tiny
    // longitude differences there at resolution 14/15 (the twin projects unit vectors directly).
    for (let row = 0; row < 120; row++)
      keys.push(
        latLngToCell(
          (Math.asin((2 * row + 1) / 120 - 1) * 180) / Math.PI,
          ((row * 137.50776405 + 0.123) % 360) - 180,
          resolution
        )
      );
    const result = computeCellMeasuresOnCPU(packKeys(keys), {family: 'h3', maximumEdgeCount: 10});
    keys.forEach((key, row) => {
      expect(result.edgeCounts[row], key).toBe(cellToBoundary(key).length);
      expect(Math.abs(result.areas[row] / cellArea(key, 'm2') - 1), key).toBeLessThan(2e-5);
    });
  }
});

it('bounds the seam-cell area independently by integrating the gnomonic area Jacobian', () => {
  const result = computeCellMeasuresOnCPU(packKeys(['8ff29380e0d0cc4']), {
    family: 'h3',
    maximumEdgeCount: 10
  });
  // Hand-decoded base cell 121 has home face 18 and axial coordinates (1, 0).
  // Its digits are 223401603206304. Alternating axial aperture-7 matrices and adding
  // their unit steps gives (1675619, 333900); aperture-33r then aperture-7r gives
  // substrate coordinates (14078871, 7030257) at adjusted resolution 16.
  const scale = 0.381966011250105 / (3 * 7 ** 8);
  const horizontal = (14078871 - 7030257 / 2) * scale;
  const vertical = ((7030257 * Math.sqrt(3)) / 2) * scale;
  // Odd substrate vertex (5,4,0) has planar radius sqrt(21), so the regular hexagon's
  // planar area is 63 sqrt(3)/2 * scale^2. Integrate dA = (1+x^2+y^2)^(-3/2) dx dy
  // by bounding its density over this tiny hexagon, independently of triangle excess.
  const estimatedArea =
    (6371007.180918475 ** 2 * ((63 * Math.sqrt(3)) / 2) * scale ** 2) /
    (1 + horizontal ** 2 + vertical ** 2) ** 1.5;
  // |gradient(log density)| <= 3/2 everywhere. Its variation over radius sqrt(21)*scale
  // is < 1.6e-7 relative; 5e-7 also covers the final float32 rounding and coordinate error.
  expect(Math.abs(result.areas[0] / estimatedArea - 1)).toBeLessThan(5e-7);
  expect(estimatedArea).toBeCloseTo(0.9725676637, 9);
  expect(Array.from(result.edgeCounts)).toEqual([6]);
  expect(Array.from(result.pentagons)).toEqual([0]);
});

it('preserves units, word order, truncation, invalid zeros and empty pure input', () => {
  const key = '81083ffffffffff'; // Class III pentagon: ten boundary edges.
  const meters = computeCellMeasuresOnCPU(packKeys([key]), {family: 'h3', maximumEdgeCount: 10});
  const kilometers = computeCellMeasuresOnCPU(packKeys([key], true), {
    family: 'h3',
    wordOrder: 'high-low',
    units: 'kilometers',
    maximumEdgeCount: 6
  });
  expect(meters.edgeCounts[0]).toBe(10);
  expect(kilometers.edgeCounts[0]).toBe(10);
  expect(Math.abs(meters.areas[0] / kilometers.areas[0] / 1e6 - 1)).toBeLessThan(2e-7);
  expect(Math.abs(meters.perimeters[0] / kilometers.perimeters[0] / 1000 - 1)).toBeLessThan(2e-7);
  for (let edge = 0; edge < 6; edge++)
    expect(
      Math.abs(meters.edgeLengths[edge] / kilometers.edgeLengths[edge] / 1000 - 1)
    ).toBeLessThan(2e-7);
  for (const values of Object.values(
    computeCellMeasuresOnCPU(packKeys(['0', '1', 'ffffffffffffffff', '81087ffffffffff']))
  ))
    expect(values.every(value => value === 0)).toBe(true);
  expect(computeCellMeasuresOnCPU(new Uint32Array()).areas.length).toBe(0);
  expect(() => computeCellMeasuresOnCPU(new Uint32Array(3))).toThrow(/cells/);
});

it('validates both operation constructors and deferred GPU contributor construction', () => {
  const shapes = {cells: {format: 'uint32x2' as const, length: 2}};
  expect(() =>
    constructNetworkOperationContributor(new GPUCellMeasuresOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUCellMeasuresOperation, CPUCellMeasuresOperation]) {
    expect(() => new Operation({...PROPS, rowCount: 0})).toThrow(/measure-test/);
    expect(() => new Operation({...PROPS, output: {areas: 'cells'}})).toThrow(/measure-test/);
    expect(() => new Operation({...PROPS, output: {areas: 'same', perimeters: 'same'}})).toThrow(
      /measure-test/
    );
  }
  // @ts-expect-error Deliberate empty terminal selection exercises both runtime guards.
  const empty: CellMeasuresOperationProps['output'] = {};
  for (const Operation of [GPUCellMeasuresOperation, CPUCellMeasuresOperation]) {
    expect(() => new Operation({...PROPS, output: empty})).toThrow(/measure-test.*at least one/);
  }
  for (const changes of [
    {maximumEdgeCount: 5},
    {maximumEdgeCount: 11},
    {maximumEdgeCount: 6.5},
    {family: 'geohash' as 'h3'},
    {units: 'native' as 'meters'},
    {wordOrder: 'wrong' as 'high-low'}
  ]) {
    expect(() => new CPUCellMeasuresOperation({...PROPS, ...changes})).toThrow(/measure-test/);
    expect(() =>
      constructNetworkOperationContributor(
        new GPUCellMeasuresOperation({...PROPS, ...changes}),
        shapes
      )
    ).toThrow(/measure-test/);
  }
});
