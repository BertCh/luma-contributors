// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNormalizePolygonsOnCPU,
  CPUNormalizePolygonsOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation';
import {
  GPUNormalizePolygonsOperation,
  type NormalizePolygonsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons-operation';
import fixture from './normalize-polygons-fixture.json';

it('matches an independent GEOS fixture with multiple features, polygon parts and holes', () => {
  const input = {
    positions: new Float32Array(fixture.input.positions),
    ringOffsets: new Uint32Array(fixture.input.ringOffsets),
    polygonOffsets: new Uint32Array(fixture.input.polygonOffsets),
    featureOffsets: new Uint32Array(fixture.input.featureOffsets)
  };
  const result = computeNormalizePolygonsOnCPU(input);
  expect(Array.from(result.positions)).toEqual(fixture.expected.positions);
  expect(Array.from(result.ringOffsets)).toEqual(fixture.expected.ringOffsets);
  expect(Array.from(result.polygonOffsets)).toEqual(fixture.expected.polygonOffsets);
  expect(computeNormalizePolygonsOnCPU({...input, ...result})).toEqual(result);
});

it('preserves empty parts, closure conventions, duplicate minima and zero-winding direction', () => {
  const result = computeNormalizePolygonsOnCPU({
    positions: new Float32Array([2, 0, 0, 0, 1, 0, 0, 0, -0, -0]),
    ringOffsets: new Uint32Array([0, 4, 4, 5]),
    polygonOffsets: new Uint32Array([0, 2, 2, 3]),
    featureOffsets: new Uint32Array([0, 3])
  });
  expect(Array.from(result.positions)).toEqual([0, 0, 1, 0, 0, 0, 2, 0, 0, 0]);
  expect(Array.from(result.ringOffsets)).toEqual([0, 4, 4, 5]);
  expect(Array.from(result.polygonOffsets)).toEqual([0, 2, 3, 3]);
  expect(result.positions.every(value => !Object.is(value, -0))).toBe(true);
});

it('rejects malformed arrays and validates both operation contracts', () => {
  const input = {
    positions: new Float32Array([0, 0]),
    ringOffsets: new Uint32Array([0, 1]),
    polygonOffsets: new Uint32Array([0, 1]),
    featureOffsets: new Uint32Array([0, 1])
  };
  expect(() => computeNormalizePolygonsOnCPU({...input, positions: new Float32Array(1)})).toThrow(
    /positions/
  );
  for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const)
    expect(() =>
      computeNormalizePolygonsOnCPU({...input, [name]: new Uint32Array([1, 0])})
    ).toThrow(new RegExp(name));
  const props: NormalizePolygonsOperationProps = {
    id: 'polygon-check',
    positions: 'positions',
    ringOffsets: 'rings',
    polygonOffsets: 'polygons',
    featureOffsets: 'features',
    rowCount: 1,
    ringCount: 1,
    polygonCount: 1,
    featureCount: 1,
    output: {
      positions: 'normalized',
      ringOffsets: 'normalizedRings',
      polygonOffsets: 'normalizedPolygons'
    }
  };
  for (const Operation of [CPUNormalizePolygonsOperation, GPUNormalizePolygonsOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    // rowCount 0 is a canonical empty source (DGN-09); a negative row count is not.
    expect(() => new Operation({...props, rowCount: -1})).toThrow(/^polygon-check/);
    expect(
      () => new Operation({...props, output: {...props.output, positions: 'positions'}})
    ).toThrow(/^polygon-check/);
  }
  expect(() => new CPUNormalizePolygonsOperation({...props, ringCount: -1})).toThrow(
    /^polygon-check.*ringCount/
  );
});
