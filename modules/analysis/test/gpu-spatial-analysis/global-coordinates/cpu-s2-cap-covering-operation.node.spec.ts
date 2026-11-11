// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getS2IndexFromCell} from '@math.gl/dggs';
import {
  computeS2CapCoveringOnCPU,
  CPUS2CapCoveringOperation
} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {
  GPUS2CapCoveringOperation,
  type S2CapCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import fixtures from './s2-cap-covering-operation-fixture.json';

const PROPS: S2CapCoveringOperationProps = {
  id: 'cap-test',
  caps: 'caps',
  capCount: 2,
  capacity: 6,
  level: 2,
  output: {cells: 'cells', counts: 'counts', status: 'status'}
};

it('matches independent SciPy edge optimization on every selected key, including overflow', () => {
  for (const fixture of fixtures)
    for (const capacity of [3, 6 * 4 ** fixture.level]) {
      const result = computeS2CapCoveringOnCPU(Float32Array.from(fixture.caps), {
        ...fixture,
        mode: fixture.mode as 'exterior' | 'interior',
        capacity
      });
      fixture.cells.forEach((expected, cap) => {
        const written = Math.min(capacity, expected.length);
        expect(Array.from(result.counts.subarray(cap * 2, cap * 2 + 2))).toEqual([
          written,
          expected.length
        ]);
        expect(result.status[cap]).toBe(expected.length > capacity ? 2 : 0);
        for (let row = 0; row < capacity; row++) {
          const offset = (cap * capacity + row) * 2;
          const cell = BigInt(result.cells[offset]) | (BigInt(result.cells[offset + 1]) << 32n);
          expect(cell, `level ${fixture.level} ${fixture.mode} cap ${cap} row ${row}`).toBe(
            row < written ? BigInt(`0x${expected[row]}`) : 0n
          );
        }
      });
    }
});

it('uses closed-form face-center and whole-sphere covers with correct clamping and zero fill', () => {
  const point = computeS2CapCoveringOnCPU(Float32Array.of(1, 0, 0, 0), {level: 0, capacity: 6});
  expect(Array.from(point.counts)).toEqual([1, 1]);
  expect(Array.from(point.cells)).toEqual([0, 0x10000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const full = computeS2CapCoveringOnCPU(Float32Array.of(1, 0, 0, Math.PI), {
    level: 12,
    capacity: 3,
    refinement: 'level-synchronous',
    frontierCapacity: 6
  });
  expect(Array.from(full.counts)).toEqual([3, 6 * 4 ** 12]);
  expect(full.status[0]).toBe(2);
  // First three leaf IDs of face zero; direct subtree counting still writes their exact IDs.
  expect(Array.from(full.cells)).toEqual([0, 16, 0, 48, 0, 80]);
  expect(() =>
    computeS2CapCoveringOnCPU(Float32Array.of(1, 0, 0, Math.PI), {level: 30, capacity: 1})
  ).toThrow(/requiredCount/);
  expect(
    computeS2CapCoveringOnCPU(Float32Array.of(1, 0, 0, 0), {
      level: 3,
      capacity: 1,
      mode: 'interior'
    }).counts[1]
  ).toBe(0);
});

it('reports invalid caps as rows, and rejects incomplete packed arrays', () => {
  const result = computeS2CapCoveringOnCPU(
    Float32Array.of(0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 4, NaN, 0, 0, 1, 1, 0, 0, Infinity),
    {level: 1, capacity: 3}
  );
  expect(Array.from(result.status)).toEqual([1, 1, 1, 1, 1]);
  expect(result.cells.every(value => value === 0)).toBe(true);
  expect(result.counts.every(value => value === 0)).toBe(true);
  expect(() => computeS2CapCoveringOnCPU(new Float32Array(3), PROPS)).toThrow(/caps/);
  expect(computeS2CapCoveringOnCPU(new Float32Array(), PROPS).counts.length).toBe(0);
});

it('resolves the four cells around a face-center point at level twenty', () => {
  // The S2 quadratic step next to the face center is >1.27e-6 radians at level 20,
  // so the 1e-6 exterior margin reaches precisely the four cells sharing that point.
  const level = 20;
  const result = computeS2CapCoveringOnCPU(Float32Array.of(1, 0, 0, 0), {
    level,
    capacity: 4,
    tolerance: 1e-6
  });
  expect(Array.from(result.counts)).toEqual([4, 4]);
  expect(result.status[0]).toBe(0);
  const half = 2 ** (level - 1);
  const expected = [
    getS2IndexFromCell({face: 0, ij: [half - 1, half - 1], level}),
    getS2IndexFromCell({face: 0, ij: [half, half - 1], level}),
    getS2IndexFromCell({face: 0, ij: [half - 1, half], level}),
    getS2IndexFromCell({face: 0, ij: [half, half], level})
  ].sort((first, second) => (first < second ? -1 : first > second ? 1 : 0));
  const actual = Array.from(
    {length: 4},
    (_, row) => BigInt(result.cells[row * 2]) | (BigInt(result.cells[row * 2 + 1]) << 32n)
  );
  expect(actual).toEqual(expected);
});

it('validates both constructions and drops only GPU traversal limits', () => {
  const shapes = {caps: {format: 'float32x4' as const, length: 2}};
  expect(() =>
    constructNetworkOperationContributor(new GPUS2CapCoveringOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation]) {
    expect(() => new Operation({...PROPS, capCount: 0})).toThrow(/cap-test/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, cells: 'caps'}})).toThrow(
      /cap-test/
    );
  }
  for (const level of [-1, 0.5, NaN]) {
    expect(() => new CPUS2CapCoveringOperation({...PROPS, level})).toThrow(/cap-test.*level/);
    expect(() =>
      constructNetworkOperationContributor(new GPUS2CapCoveringOperation({...PROPS, level}), shapes)
    ).toThrow(/cap-test.*level/);
  }
  expect(
    () => new CPUS2CapCoveringOperation({...PROPS, level: 30, frontierCapacity: 1})
  ).not.toThrow();
  expect(() => new CPUS2CapCoveringOperation({...PROPS, level: 31})).toThrow(/level/);
  for (const tolerance of [0, NaN, Infinity, 0.1]) {
    expect(() => new CPUS2CapCoveringOperation({...PROPS, tolerance})).toThrow(/tolerance/);
    expect(() =>
      constructNetworkOperationContributor(
        new GPUS2CapCoveringOperation({...PROPS, tolerance}),
        shapes
      )
    ).toThrow(/tolerance/);
  }
});
