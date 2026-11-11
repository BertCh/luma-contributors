// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeS2PolygonCoveringOnCPU,
  CPUS2PolygonCoveringOperation
} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {
  GPUS2PolygonCoveringOperation,
  type S2PolygonCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import fixtures from './s2-polygon-covering-operation-fixture.json';

const PROPS: S2PolygonCoveringOperationProps = {
  id: 'polygon-test',
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  positionCount: 4,
  ringOffsetCount: 2,
  maximumLevel: 3,
  capacity: 128,
  output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visitedCount'}
};

it('matches independent Shapely gnomonic intersections: holes, complement, adaptive and uniform covers', () => {
  for (const fixture of fixtures)
    for (const capacity of [3, 512]) {
      const result = computeS2PolygonCoveringOnCPU(
        {
          positions: Float32Array.from(fixture.positions),
          ringOffsets: Uint32Array.from(fixture.ringOffsets)
        },
        {...fixture, mode: fixture.mode as 'exterior' | 'interior', capacity}
      );
      const written = Math.min(capacity, fixture.cells.length);
      expect(Array.from(result.counts), fixture.name).toEqual([written, fixture.cells.length]);
      expect(result.status[0], fixture.name).toBe(fixture.cells.length > capacity ? 2 : 0);
      expect(result.visitedCount[0], fixture.name).toBe(fixture.visitedCount);
      for (let row = 0; row < written; row++) {
        const cell = BigInt(result.cells[row * 2]) | (BigInt(result.cells[row * 2 + 1]) << 32n);
        expect(cell, fixture.name).toBe(BigInt(`0x${fixture.cells[row]}`));
      }
      expect(result.cells.subarray(written * 2).every(value => value === 0)).toBe(true);
    }
});

it('completes beyond the GPU visit budget and gives complete counts on output overflow', () => {
  const fixture = fixtures.find(scene => scene.minimumLevel === 3 && !scene.complement)!;
  const result = computeS2PolygonCoveringOnCPU(
    {
      positions: Float32Array.from(fixture.positions),
      ringOffsets: Uint32Array.from(fixture.ringOffsets)
    },
    {...fixture, mode: 'exterior', capacity: 1, maximumVisitedCells: 6}
  );
  expect(result.status[0]).toBe(2);
  expect(result.counts[1]).toBe(fixture.cells.length);
  expect(result.visitedCount[0]).toBeGreaterThan(6);
});

it('returns invalid status for malformed ranges, non-unit vertices, collapsed and crossing rings', () => {
  const options = {maximumLevel: 2, capacity: 4};
  const triangle = Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  for (const ringOffsets of [Uint32Array.of(0, 2), Uint32Array.of(3, 1), Uint32Array.of(0, 4)]) {
    const result = computeS2PolygonCoveringOnCPU({positions: triangle, ringOffsets}, options);
    expect(Array.from(result.status)).toEqual([1]);
    expect(Array.from(result.counts)).toEqual([0, 0]);
    expect(Array.from(result.visitedCount)).toEqual([0]);
    expect(result.cells.every(value => value === 0)).toBe(true);
  }
  for (const positions of [
    new Float32Array(9),
    Float32Array.of(1, 0, 0, 1, 0, 0, 0, 1, 0),
    Float32Array.of(NaN, 0, 0, 0, 1, 0, 0, 0, 1)
  ])
    expect(
      computeS2PolygonCoveringOnCPU({positions, ringOffsets: Uint32Array.of(0, 3)}, options)
        .status[0]
    ).toBe(1);
  const fixture = fixtures[0];
  const crossing = Float32Array.from(
    [0, 2, 1, 3].flatMap(row => fixture.positions.slice(row * 3, row * 3 + 3))
  );
  expect(
    computeS2PolygonCoveringOnCPU({positions: crossing, ringOffsets: Uint32Array.of(0, 4)}, options)
      .status[0]
  ).toBe(1);
});

it('validates arrays and both constructions, retaining only layout limits on the CPU', () => {
  expect(() =>
    computeS2PolygonCoveringOnCPU(
      {positions: new Float32Array(2), ringOffsets: Uint32Array.of(0, 0)},
      PROPS
    )
  ).toThrow(/positions/);
  expect(() =>
    computeS2PolygonCoveringOnCPU(
      {positions: new Float32Array(0), ringOffsets: new Uint32Array(1)},
      PROPS
    )
  ).toThrow(/ringOffsets/);
  const shapes = {
    positions: {format: 'float32x3' as const, length: 4},
    ringOffsets: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUS2PolygonCoveringOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation]) {
    expect(() => new Operation({...PROPS, capacity: 0})).toThrow(/polygon-test.*length/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, cells: 'positions'}})).toThrow(
      /polygon-test/
    );
  }
  for (const maximumLevel of [-1, 0.5, NaN]) {
    expect(() => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel})).toThrow(
      /polygon-test.*maximumLevel/
    );
    expect(() =>
      constructNetworkOperationContributor(
        new GPUS2PolygonCoveringOperation({...PROPS, maximumLevel}),
        shapes
      )
    ).toThrow(/polygon-test.*maximumLevel/);
  }
  expect(
    () => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel: 30, maximumVisitedCells: 1})
  ).not.toThrow();
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, maximumLevel: 31})).toThrow(
    /maximumLevel/
  );
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, minimumLevel: 4})).toThrow(
    /minimumLevel/
  );
  expect(() =>
    constructNetworkOperationContributor(
      new GPUS2PolygonCoveringOperation({...PROPS, minimumLevel: 4}),
      shapes
    )
  ).toThrow(/minimumLevel/);
  expect(() => new CPUS2PolygonCoveringOperation({...PROPS, hemisphereCenter: [0, 0, 0]})).toThrow(
    /hemisphereCenter/
  );
  expect(() =>
    constructNetworkOperationContributor(
      new GPUS2PolygonCoveringOperation({...PROPS, hemisphereCenter: [0, 0, 0]}),
      shapes
    )
  ).toThrow(/hemisphereCenter/);
});
