// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeLabelPointOnCPU,
  CPULabelPointOperation
} from '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';
import {
  GPULabelPointOperation,
  type LabelPointOperationProps
} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point-operation';
import {flattenPolygonFeatures} from '../geometry-measures/polygon-layout';

const PROPS: LabelPointOperationProps = {
  id: 'label-test',
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  polygonOffsets: 'polygonOffsets',
  featureOffsets: 'featureOffsets',
  featureCount: 1,
  output: {points: 'points', distances: 'distances', degenerate: 'degenerate'}
};

it('recovers a 3-4-5 triangle incenter (closed form), independent of winding and translation', () => {
  for (const reverse of [false, true]) {
    const ring = [
      [100, 200],
      [103, 200],
      [100, 204]
    ];
    const result = computeLabelPointOnCPU(
      flattenPolygonFeatures([[reverse ? ring.reverse() : ring]])
    );
    expect(result.points[0]).toBeCloseTo(101, 4);
    expect(result.points[1]).toBeCloseTo(201, 4);
    expect(result.distances[0]).toBeCloseTo(1, 4);
    expect(Array.from(result.degenerate)).toEqual([0]);
  }
});

it('finds the analytic pole of a square annulus, outside its hole', () => {
  const input = flattenPolygonFeatures([
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
    ]
  ]);
  const result = computeLabelPointOnCPU(input);
  const radius = 2 - Math.SQRT2;
  expect(result.distances[0]).toBeCloseTo(radius, 4);
  expect(Math.min(result.points[0], 4 - result.points[0])).toBeCloseTo(radius, 4);
  expect(Math.min(result.points[1], 4 - result.points[1])).toBeCloseTo(radius, 4);
});

it('searches all parts of a multipolygon and emits every feature row', () => {
  const input = {
    positions: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 10, 0, 14, 0, 14, 4, 10, 4]),
    ringOffsets: new Uint32Array([0, 4, 8]),
    polygonOffsets: new Uint32Array([0, 1, 2]),
    featureOffsets: new Uint32Array([0, 2, 2])
  };
  const result = computeLabelPointOnCPU(input);
  expect(result.points[0]).toBeCloseTo(12, 4);
  expect(result.points[1]).toBeCloseTo(2, 4);
  expect(result.distances[0]).toBeCloseTo(2, 4);
  expect(Number.isNaN(result.distances[1])).toBe(true);
  expect(Array.from(result.degenerate)).toEqual([0, 1]);
});

it('keeps the first scanline seed when two polygon parts have equal poles', () => {
  const result = computeLabelPointOnCPU({
    positions: new Float32Array([0, 0, 2, 0, 2, 2, 0, 2, 4, 0, 6, 0, 6, 2, 4, 2]),
    ringOffsets: new Uint32Array([0, 4, 8]),
    polygonOffsets: new Uint32Array([0, 1, 2]),
    featureOffsets: new Uint32Array([0, 2])
  });
  expect(Array.from(result.points)).toEqual([1, 1]);
  expect(Array.from(result.distances)).toEqual([1]);
});

it('reports empty, non-finite and collapsed features and names inconsistent arguments', () => {
  const input = flattenPolygonFeatures([
    [],
    [
      [
        [2, 3],
        [2, 7]
      ]
    ],
    [[[NaN, 0]]]
  ]);
  const result = computeLabelPointOnCPU(input);
  expect(Array.from(result.points)).toEqual([NaN, NaN, 2, 3, NaN, NaN]);
  expect(Array.from(result.distances)).toEqual([NaN, 0, NaN]);
  expect(Array.from(result.degenerate)).toEqual([1, 1, 1]);
  expect(() => computeLabelPointOnCPU({...input, positions: new Float32Array(1)})).toThrow(
    /positions/
  );
  for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
    expect(() => computeLabelPointOnCPU({...input, [name]: new Uint32Array(1)})).toThrow(name);
  }
});

it('validates operation output shape and aliases at construction, and drops GPU-only limits', () => {
  for (const Operation of [GPULabelPointOperation, CPULabelPointOperation]) {
    expect(() => new Operation(PROPS)).not.toThrow();
    expect(() => new Operation({...PROPS, featureCount: 0})).toThrow(
      /^label-test.*length|^label-test.*featureCount/
    );
    expect(() => new Operation({...PROPS, output: {...PROPS.output, points: 'positions'}})).toThrow(
      /^label-test/
    );
  }
  expect(
    () => new CPULabelPointOperation({...PROPS, refinementCandidates: 1000, refinementRounds: 1000})
  ).not.toThrow();
  expect(
    () =>
      new CPULabelPointOperation({
        ...PROPS,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/^label-test/);
});

it('uses the empty feature interval when the final ring offset descends to the first row', () => {
  const result = computeLabelPointOnCPU({
    positions: new Float32Array([0, 0, 4, 0, 4, 4, 0, 4]),
    ringOffsets: new Uint32Array([0, 4, 0]),
    polygonOffsets: new Uint32Array([0, 2]),
    featureOffsets: new Uint32Array([0, 1])
  });
  // The overall interval is [ringOffsets[0], ringOffsets[2]) = [0, 0), despite ring 0.
  expect(Array.from(result.points)).toEqual([NaN, NaN]);
  expect(Array.from(result.distances)).toEqual([NaN]);
  expect(Array.from(result.degenerate)).toEqual([1]);
});
