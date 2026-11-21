// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMaximumInscribedCircleOnCPU,
  CPUMaximumInscribedCircleOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {
  GPUMaximumInscribedCircleOperation,
  type MaximumInscribedCircleOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle-operation';
import {flattenPolygonFeatures} from '../geometry-measures/polygon-layout';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';

const PROPS: MaximumInscribedCircleOperationProps = {
  id: 'circle-test',
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  polygonOffsets: 'polygonOffsets',
  featureOffsets: 'featureOffsets',
  featureCount: 1,
  tolerance: 1e-5,
  output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
};

it('recovers the closed-form square circle and first closest-edge witness', () => {
  const result = computeMaximumInscribedCircleOnCPU(
    flattenPolygonFeatures([
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ]
      ]
    ]),
    1e-5
  );
  expect(Array.from(result.circles.slice(0, 3))).toEqual([2, 2, 2]);
  expect(result.circles[3]).toBeGreaterThanOrEqual(0);
  expect(result.circles[3]).toBeLessThanOrEqual(1e-5);
  expect(Array.from(result.witnesses)).toEqual([0, 2]);
  expect(Array.from(result.status)).toEqual([0]);
});

it('bounds the analytic 3-4-5 inradius and returns a witness on a triangle edge', () => {
  const result = computeMaximumInscribedCircleOnCPU(
    flattenPolygonFeatures([
      [
        [
          [0, 0],
          [3, 0],
          [0, 4]
        ]
      ]
    ]),
    1e-5
  );
  const [centerX, centerY, radius, error] = result.circles;
  const [witnessX, witnessY] = result.witnesses;
  expect(centerX).toBeCloseTo(1, 4);
  expect(centerY).toBeCloseTo(1, 4);
  expect(radius).toBeLessThanOrEqual(1);
  expect(radius + error + 1e-7).toBeGreaterThanOrEqual(1);
  expect(Math.hypot(centerX - witnessX, centerY - witnessY)).toBeCloseTo(radius, 6);
  expect(
    Math.min(Math.abs(witnessX), Math.abs(witnessY), Math.abs(4 * witnessX + 3 * witnessY - 12))
  ).toBeLessThan(1e-6);
});

it('uses even-odd holes and preserves empty/non-finite status and NaN fields', () => {
  const result = computeMaximumInscribedCircleOnCPU(
    flattenPolygonFeatures([
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ],
        [
          [1, 1],
          [3, 1],
          [3, 3],
          [1, 3]
        ]
      ],
      [],
      [[[Infinity, 0]]]
    ]),
    1e-5
  );
  expect(result.circles[2]).toBeCloseTo(2 - Math.SQRT2, 4);
  expect(Array.from(result.status)).toEqual([0, 2, 3]);
  expect(result.circles.slice(4).every(Number.isNaN)).toBe(true);
  expect(result.witnesses.slice(2).every(Number.isNaN)).toBe(true);
});

it('validates construction and pure-function arguments and honors explicit search budgets', () => {
  for (const Operation of [
    GPUMaximumInscribedCircleOperation,
    CPUMaximumInscribedCircleOperation
  ]) {
    expect(() => new Operation(PROPS)).not.toThrow();
    expect(() => new Operation({...PROPS, featureCount: -1})).toThrow(/^circle-test/);
    expect(() => new Operation({...PROPS, output: {...PROPS.output, status: 'circles'}})).toThrow(
      /^circle-test/
    );
  }
  expect(() => new CPUMaximumInscribedCircleOperation({...PROPS, tolerance: 0})).toThrow(
    /^circle-test.*tolerance/
  );
  const input = flattenPolygonFeatures([
    [
      [
        [0, 0],
        [4, 0],
        [0, 4]
      ]
    ]
  ]);
  expect(() =>
    computeMaximumInscribedCircleOnCPU({...input, positions: new Float32Array(1)}, 1)
  ).toThrow(/positions/);
  expect(() => computeMaximumInscribedCircleOnCPU(input, NaN)).toThrow(/tolerance/);
  const operation = new CPUMaximumInscribedCircleOperation({
    ...PROPS,
    maximumIterations: 0,
    cellCapacity: 100000
  });
  const makeRows = (
    format: GPUCPUVector['format'],
    values: Float32Array | Uint32Array,
    components = 1
  ): GPUCPUVector => ({
    format,
    length: values.length / components,
    chunks: [{values, length: values.length / components}],
    pack: () => values
  });
  const result = operation.execute({
    inputs: {
      positions: makeRows('float32x2', input.positions, 2),
      ringOffsets: makeRows('uint32', input.ringOffsets),
      polygonOffsets: makeRows('uint32', input.polygonOffsets),
      featureOffsets: makeRows('uint32', input.featureOffsets)
    }
  });
  expect(result).not.toBeInstanceOf(Promise);
  expect((result as Record<string, Uint32Array>)['status'][0]).toBe(1);
});

it('reports the hand-derived successful square error at tolerance 0.01', () => {
  const result = computeMaximumInscribedCircleOnCPU(
    flattenPolygonFeatures([
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ]
      ]
    ]),
    0.01
  );
  // An inscribed disk cannot exceed half the shorter envelope side, 2. The center attains it.
  expect(Array.from(result.circles.slice(0, 3))).toEqual([2, 2, 2]);
  expect(result.circles[3]).toBe(0);
  expect(result.circles[3]).toBeGreaterThanOrEqual(0);
  expect(result.circles[3]).toBeLessThanOrEqual(Math.fround(0.01));
  expect(Array.from(result.witnesses)).toEqual([0, 2]);
  expect(Array.from(result.status)).toEqual([0]);
});

it('improves the isosceles triangle seed and bounds its closed-form inradius', () => {
  const result = computeMaximumInscribedCircleOnCPU(
    flattenPolygonFeatures([
      [
        [
          [0, 0],
          [4, 0],
          [0, 4]
        ]
      ]
    ]),
    0.01
  );
  const [centerX, centerY, radius, error] = result.circles;
  // The center (2, 2) starts on x+y=4; the incenter satisfies r = (4-2r)/sqrt(2).
  const inradius = 4 - 2 * Math.SQRT2;
  expect(centerX).toBeCloseTo(inradius, 2);
  expect(centerY).toBeCloseTo(inradius, 2);
  expect(radius).toBeGreaterThan(1);
  expect(radius).toBeLessThanOrEqual(inradius);
  expect(error).toBeGreaterThanOrEqual(0);
  expect(error).toBeLessThanOrEqual(Math.fround(0.01));
  expect(radius + error + 1e-7).toBeGreaterThanOrEqual(inradius);
  expect(Math.hypot(centerX - result.witnesses[0], centerY - result.witnesses[1])).toBeCloseTo(
    radius,
    6
  );
  expect(Array.from(result.status)).toEqual([0]);
});
