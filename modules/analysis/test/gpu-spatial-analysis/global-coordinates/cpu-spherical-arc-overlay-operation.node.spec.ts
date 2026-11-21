// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {
  computeSphericalArcOverlayOnCPU,
  CPUSphericalArcOverlayOperation,
  type CPUSphericalArcOverlayInput,
  type CPUSphericalArcOverlayOptions,
  type CPUSphericalArcOverlayResult
} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {
  GPUSphericalArcOverlayOperation,
  type SphericalArcOverlayOperationProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import reference from './fixtures/spherical-arc-overlay.json';

const OPTIONS: CPUSphericalArcOverlayOptions = {
  referencePoint: [0, 0, -1],
  operation: 'intersection',
  capacity: 64,
  ringCapacity: 16
};
const PROPS: SphericalArcOverlayOperationProps = {
  id: 'arc-test',
  ...OPTIONS,
  left: {
    positions: 'leftPositions',
    ringOffsets: 'leftOffsets',
    referenceInside: false
  },
  right: {
    positions: 'rightPositions',
    ringOffsets: 'rightOffsets',
    referenceInside: false
  },
  output: {positions: 'positions', ringOffsets: 'ringOffsets', counts: 'counts', status: 'status'}
};

function projectRing(points: number[][]): Float32Array {
  return Float32Array.from(
    points.flatMap(([horizontal, vertical]) => {
      const length = Math.hypot(horizontal, vertical, 1);
      return [horizontal / length, vertical / length, 1 / length];
    })
  );
}
function rectangle(left: number, bottom: number, right: number, top: number): number[][] {
  return [
    [left, bottom],
    [right, bottom],
    [right, top],
    [left, top]
  ];
}
function getOperand(rings: number[][][]) {
  const offsets = [0];
  for (const ring of rings) offsets.push(offsets[offsets.length - 1] + ring.length);
  return {
    positions: projectRing(rings.flat()),
    ringOffsets: Uint32Array.from(offsets),
    referenceInside: false
  };
}
function getInput(): CPUSphericalArcOverlayInput {
  return {
    left: getOperand([rectangle(-0.6, -0.4, 0.4, 0.5)]),
    right: getOperand([rectangle(-0.2, -0.6, 0.7, 0.2)])
  };
}
// Inverse gnomonic projection turns the output great-circle edges back into straight lines.
// Shoelace area in this chart is an independent planar oracle, not spherical area arithmetic.
function getChartArea(result: CPUSphericalArcOverlayResult): number {
  let twiceArea = 0;
  for (let ring = 0; ring < result.counts[1]; ring++) {
    const begin = result.ringOffsets[ring];
    const end = result.ringOffsets[ring + 1];
    for (let vertex = begin; vertex < end; vertex++) {
      const next = vertex + 1 === end ? begin : vertex + 1;
      const positions = result.positions;
      twiceArea +=
        (positions[vertex * 3] * positions[next * 3 + 1] -
          positions[vertex * 3 + 1] * positions[next * 3]) /
        (positions[vertex * 3 + 2] * positions[next * 3 + 2]);
    }
  }
  return twiceArea / 2;
}

it('matches hand-computed Boolean rectangle areas and actual noded vertices in a gnomonic chart', () => {
  for (const [operation, area, vertices, rings] of [
    ['intersection', 0.36, 4, 1],
    ['union', 1.26, 8, 1],
    ['difference', 0.54, 6, 1],
    ['symmetric-difference', 0.9, 12, 2]
  ] as const) {
    const result = computeSphericalArcOverlayOnCPU(getInput(), {...OPTIONS, operation});
    expect(Array.from(result.status), operation).toEqual([0]);
    expect(Array.from(result.counts), operation).toEqual([vertices, rings, vertices, 0]);
    expect(getChartArea(result), operation).toBeCloseTo(area, 6);
    expect(result.positions.subarray(vertices * 3).every(value => value === 0)).toBe(true);
    expect(Array.from(result.ringOffsets.subarray(rings + 1))).toEqual(
      new Array(16 - rings).fill(vertices)
    );
    if (operation === 'intersection') {
      const actual = Array.from({length: vertices}, (_, row) => [
        result.positions[row * 3] / result.positions[row * 3 + 2],
        result.positions[row * 3 + 1] / result.positions[row * 3 + 2]
      ]);
      for (const point of rectangle(-0.2, -0.4, 0.4, 0.2))
        expect(
          actual.some(
            candidate => Math.hypot(candidate[0] - point[0], candidate[1] - point[1]) < 1e-7
          )
        ).toBe(true);
    }
  }
});

it('matches direct S2 BooleanOperation fixtures beyond a hemisphere and for complements', () => {
  for (const scene of reference.cases) {
    const result = computeSphericalArcOverlayOnCPU(
      {
        left: {
          positions: Float32Array.from(reference.left.flat()),
          ringOffsets: Uint32Array.of(0, 4),
          referenceInside: scene.leftInside
        },
        right: {
          positions: Float32Array.from(reference.right.flat()),
          ringOffsets: Uint32Array.of(0, 4),
          referenceInside: scene.rightInside
        }
      },
      {
        ...OPTIONS,
        referencePoint: [0, 0, 1],
        operation: scene.operation as CPUSphericalArcOverlayOptions['operation']
      }
    );
    expect(Array.from(result.status), JSON.stringify(scene)).toEqual([0]);
    expect(result.counts[1]).toBe(scene.rings);
    expect(result.counts[3]).toBe(Number(scene.referenceInside));
    // Signed triangle fan around the north pole measures the returned boundary. The oracle
    // is the independent committed S2 area; no overlay decision is reused to obtain it.
    let area = 0;
    for (let ring = 0; ring < result.counts[1]; ring++)
      for (let vertex = result.ringOffsets[ring]; vertex < result.ringOffsets[ring + 1]; vertex++) {
        const next =
          vertex + 1 === result.ringOffsets[ring + 1] ? result.ringOffsets[ring] : vertex + 1;
        const first = result.positions.subarray(vertex * 3, vertex * 3 + 3);
        const second = result.positions.subarray(next * 3, next * 3 + 3);
        area +=
          2 *
          Math.atan2(
            first[0] * second[1] - first[1] * second[0],
            1 +
              first[2] +
              second[2] +
              first.reduce((sum, value, axis) => sum + value * second[axis], 0)
          );
      }
    if (area < 0) area += 4 * Math.PI;
    expect(Math.abs(area - scene.area), JSON.stringify(scene)).toBeLessThan(2e-6);
  }
});

it('honors parity holes, coincident/reversed boundaries and disjoint rings', () => {
  const input = getInput();
  input.left = getOperand([rectangle(-0.8, -0.8, 0.8, 0.8), rectangle(-0.2, -0.2, 0.2, 0.2)]);
  input.right = getOperand([rectangle(0, -0.5, 0.6, 0.5)]);
  const holeResult = computeSphericalArcOverlayOnCPU(input, OPTIONS);
  expect(Array.from(holeResult.status)).toEqual([0]);
  expect(getChartArea(holeResult)).toBeCloseTo(0.52, 6);
  const square = rectangle(-0.6, -0.4, 0.4, 0.5);
  for (const operation of [
    'intersection',
    'union',
    'difference',
    'symmetric-difference'
  ] as const) {
    const result = computeSphericalArcOverlayOnCPU(
      {left: getOperand([square]), right: getOperand([[...square].reverse()])},
      {...OPTIONS, operation}
    );
    expect(Array.from(result.status), operation).toEqual([0]);
    expect(result.counts[0]).toBe(operation === 'intersection' || operation === 'union' ? 4 : 0);
    expect(getChartArea(result)).toBeCloseTo(
      operation === 'intersection' || operation === 'union' ? 0.9 : 0,
      6
    );
  }
  const disjoint = {
    left: getOperand([rectangle(-0.8, -0.6, -0.5, -0.2), rectangle(0.1, 0.2, 0.4, 0.5)]),
    right: getOperand([rectangle(0.5, -0.6, 0.8, -0.2)])
  };
  const union = computeSphericalArcOverlayOnCPU(disjoint, {...OPTIONS, operation: 'union'});
  expect(Array.from(union.status)).toEqual([0]);
  expect(Array.from(union.counts)).toEqual([12, 3, 12, 0]);
  expect(getChartArea(union)).toBeCloseTo(0.33, 6);
});

it('publishes exact required counts with zero geometry on vertex and ring capacity failure', () => {
  const square = getOperand([rectangle(-0.6, -0.4, 0.4, 0.5)]);
  const duplicateOverflow = computeSphericalArcOverlayOnCPU(
    {left: square, right: square},
    {...OPTIONS, operation: 'union', capacity: 3}
  );
  // The fourth segment is beyond capacity, but its duplicate from the second operand must
  // still be counted only once; both forms retain it in output-independent segment scratch.
  expect(Array.from(duplicateOverflow.status)).toEqual([2]);
  expect(Array.from(duplicateOverflow.counts)).toEqual([0, 0, 4, 0]);
  const vertexFailure = computeSphericalArcOverlayOnCPU(getInput(), {
    ...OPTIONS,
    operation: 'union',
    capacity: 3
  });
  expect(Array.from(vertexFailure.status)).toEqual([2]);
  expect(Array.from(vertexFailure.counts)).toEqual([0, 0, 8, 0]);
  expect(vertexFailure.positions.every(value => value === 0)).toBe(true);
  expect(vertexFailure.ringOffsets.every(value => value === 0)).toBe(true);
  const ringFailure = computeSphericalArcOverlayOnCPU(getInput(), {
    ...OPTIONS,
    operation: 'symmetric-difference',
    ringCapacity: 1
  });
  expect(Array.from(ringFailure.status)).toEqual([2]);
  expect(Array.from(ringFailure.counts)).toEqual([0, 0, 12, 0]);
  expect(ringFailure.positions.every(value => value === 0)).toBe(true);
  expect(ringFailure.ringOffsets.every(value => value === 0)).toBe(true);
});

it('returns invalid topology statuses, rejects malformed packing, and handles empty/full spheres', () => {
  for (const offsets of [
    [1, 4],
    [0, 5],
    [0, 3, 2, 4],
    [0, 2, 4]
  ]) {
    const input = getInput();
    input.left.ringOffsets = Uint32Array.from(offsets);
    const result = computeSphericalArcOverlayOnCPU(input, OPTIONS);
    expect(Array.from(result.status)).toEqual([1]);
    expect(Array.from(result.counts)).toEqual([0, 0, 0, 0]);
  }
  for (const positions of [
    Float32Array.of(1, 0, 0, 1, 0, 0, 0, 0, 1),
    projectRing([
      [0, 0],
      [0.5, 0.5],
      [0, 0.5],
      [0.5, 0]
    ])
  ]) {
    const input = getInput();
    input.left = {
      positions,
      ringOffsets: Uint32Array.of(0, positions.length / 3),
      referenceInside: false
    };
    expect(computeSphericalArcOverlayOnCPU(input, OPTIONS).status[0]).toBe(1);
  }
  const input = getInput();
  input.left.positions[0] = NaN;
  expect(computeSphericalArcOverlayOnCPU(input, OPTIONS).status[0]).toBe(1);
  expect(() =>
    computeSphericalArcOverlayOnCPU(
      {
        ...getInput(),
        left: {
          positions: new Float32Array(2),
          ringOffsets: Uint32Array.of(0),
          referenceInside: false
        }
      },
      OPTIONS
    )
  ).toThrow(/left.positions/);
  expect(() =>
    computeSphericalArcOverlayOnCPU(
      {
        ...getInput(),
        right: {
          positions: new Float32Array(),
          ringOffsets: new Uint32Array(),
          referenceInside: false
        }
      },
      OPTIONS
    )
  ).toThrow(/right.ringOffsets/);
  for (const inside of [false, true]) {
    const empty = {
      positions: new Float32Array(),
      ringOffsets: Uint32Array.of(0),
      referenceInside: inside
    };
    const result = computeSphericalArcOverlayOnCPU({left: empty, right: empty}, OPTIONS);
    expect(Array.from(result.status)).toEqual([0]);
    expect(Array.from(result.counts)).toEqual([0, 0, 0, Number(inside)]);
  }
});

it('drops GPU input and output caps while preserving the uint32 layout', () => {
  const polygon = (center: number) =>
    Array.from({length: 65}, (_, vertex) => [
      center + 0.1 * Math.cos((vertex * 2 * Math.PI) / 65),
      0.1 * Math.sin((vertex * 2 * Math.PI) / 65)
    ]);
  const result = computeSphericalArcOverlayOnCPU(
    {left: getOperand([polygon(-0.4)]), right: getOperand([polygon(0.4)])},
    {...OPTIONS, operation: 'union', capacity: 1025}
  );
  expect(Array.from(result.status)).toEqual([0]);
  expect(Array.from(result.counts)).toEqual([130, 2, 130, 0]);
  expect(
    () =>
      new CPUSphericalArcOverlayOperation({
        ...PROPS,
        capacity: 1025
      })
  ).not.toThrow();
});

it('validates both operation classes, with GPU geometry checks at deferred construction', () => {
  expect(
    () => new CPUSphericalArcOverlayOperation({...PROPS, operation: 'invalid' as 'union'})
  ).toThrow(/arc-test.*operation/);
  const shapes = {
    leftPositions: {format: 'float32x3' as const, length: 4},
    rightPositions: {format: 'float32x3' as const, length: 4},
    leftRingOffsets: {format: 'uint32' as const, length: 2},
    rightRingOffsets: {format: 'uint32' as const, length: 2}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPUSphericalArcOverlayOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPUSphericalArcOverlayOperation, CPUSphericalArcOverlayOperation]) {
    expect(() => new Operation({...PROPS, capacity: 0})).toThrow(/arc-test/);
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, positions: 'leftPositions'}})
    ).toThrow(/arc-test/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, status: 'counts'}})).toThrow(
      /arc-test/
    );
  }
  for (const changes of [
    {tolerance: 0},
    {tolerance: NaN},
    {referencePoint: [0, 0, 0] as const},
    {capacity: 2},
    {ringCapacity: 0},
    {spatialContext: {coordinateSpace: 'planar' as const, metric: 'native' as const}}
  ]) {
    expect(() => new CPUSphericalArcOverlayOperation({...PROPS, ...changes})).toThrow(/arc-test/);
    expect(() =>
      constructNetworkOperationContributor(
        new GPUSphericalArcOverlayOperation({...PROPS, ...changes}),
        shapes
      )
    ).toThrow(/arc-test/);
  }
  expect(() =>
    constructNetworkOperationContributor(
      new GPUSphericalArcOverlayOperation({...PROPS, capacity: 1025}),
      shapes
    )
  ).toThrow(/arc-test/);
  expect(() =>
    constructNetworkOperationContributor(new GPUSphericalArcOverlayOperation(PROPS), {
      ...shapes,
      leftPositions: {format: 'float32x3', length: 125}
    })
  ).toThrow(/arc-test/);
});

it('derives operation input lengths from vectors and preserves malformed-array checks', async () => {
  const square = getOperand([rectangle(-0.6, -0.4, 0.4, 0.5)]);
  const makeVector = (values: Float32Array | Uint32Array): GPUCPUVector => {
    const format = values instanceof Float32Array ? 'float32x3' : 'uint32';
    const length = values.length / (format === 'float32x3' ? 3 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const inputs = {
    leftPositions: makeVector(square.positions),
    leftRingOffsets: makeVector(square.ringOffsets),
    rightPositions: makeVector(square.positions),
    rightRingOffsets: makeVector(square.ringOffsets)
  };
  const shapes = {
    leftPositions: {format: 'float32x3' as const, length: 4},
    rightPositions: {format: 'float32x3' as const, length: 4},
    leftRingOffsets: {format: 'uint32' as const, length: 2},
    rightRingOffsets: {format: 'uint32' as const, length: 2}
  };
  // Extra metadata from the previous API is ignored on both paths. These are exactly the
  // two reported reproductions; the public props type no longer asks callers for these counts.
  for (const vertexCount of [1, 4]) {
    const props = {...PROPS, left: {...PROPS.left, vertexCount, ringOffsetCount: 1}};
    expect(() =>
      constructNetworkOperationContributor(new GPUSphericalArcOverlayOperation(props), shapes)
    ).not.toThrow();
    const operation = new CPUSphericalArcOverlayOperation(props);
    const result = await operation.execute({inputs});
    expect(Array.from(result['counts'] as Uint32Array)).toEqual([4, 1, 4, 0]);
    expect(Array.from(result['status'] as Uint32Array)).toEqual([0]);
    // A square intersected with itself has its four original vertices, in source order.
    expect(Array.from((result['positions'] as Float32Array).subarray(0, 12))).toEqual(
      Array.from(square.positions)
    );
  }
  const operation = new CPUSphericalArcOverlayOperation(PROPS);
  expect(() =>
    operation.execute({inputs: {...inputs, leftPositions: makeVector(new Float32Array(2))}})
  ).toThrow(/left.positions/);
  expect(() =>
    operation.execute({inputs: {...inputs, leftRingOffsets: makeVector(new Uint32Array())}})
  ).toThrow(/left.ringOffsets/);
  const invalid = await operation.execute({
    inputs: {...inputs, leftRingOffsets: makeVector(Uint32Array.of(0, 5))}
  });
  expect(Array.from(invalid['status'] as Uint32Array)).toEqual([1]);
});
