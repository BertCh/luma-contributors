// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {PAIR_STATISTICS_BOUNDARY_SECOND_POINT} from './pair-statistics-boundary-fixture';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';

it('DPC-108 rejects overflowing float32 extents and duration despite finite ordered endpoints', () => {
  const settings = {
    bounds: [0, 0, 1, 1] as const,
    timeWindow: [0, 1] as const,
    maximumDistance: 1,
    maximumTimeLag: 1
  };
  // Both endpoints round to finite f32 values, but their f32 difference is infinite.
  expect(Number.isFinite(Math.fround(2e38))).toBe(true);
  expect(Math.fround(Math.fround(2e38) - Math.fround(-2e38))).toBe(Number.POSITIVE_INFINITY);
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({...settings, bounds: [-2e38, 0, 2e38, 1]})
  ).toThrow(
    'getGPUSpaceTimeKFunctionParameterValues bounds width must be positive and finite in float32'
  );
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({...settings, bounds: [0, -2e38, 1, 2e38]})
  ).toThrow(
    'getGPUSpaceTimeKFunctionParameterValues bounds height must be positive and finite in float32'
  );
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({...settings, timeWindow: [-2e38, 2e38]})
  ).toThrow(
    'getGPUSpaceTimeKFunctionParameterValues timeWindow duration must be positive and finite in float32'
  );
  // Width/duration are finite, but their product exceeds the shader's f32 range.
  expect(Math.fround(Math.fround(2e38) * Math.fround(2e38))).toBe(Infinity);
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({
      ...settings,
      bounds: [-1e38, 0, 1e38, 1],
      timeWindow: [-1e38, 1e38]
    })
  ).toThrow(/space-time volume/);
});

it('S2 the boundary fixture distinguishes the polygon reverse arc from the bounding rectangle', () => {
  const coordinate = -PAIR_STATISTICS_BOUNDARY_SECOND_POINT[0];
  expect(PAIR_STATISTICS_BOUNDARY_SECOND_POINT[1]).toBe(-coordinate);
  const radius = Math.SQRT2 * coordinate;
  const hypotenuseDistance = (1 - 2 * coordinate) / Math.SQRT2;
  // The axis cuts have half-angle pi/4 each and do not overlap. The hypotenuse's
  // disjoint cut has half-angle acos(h/d); subtract its fraction from the remaining half.
  const reverseFraction = 0.5 - Math.acos(Math.min(1, hypotenuseDistance / radius)) / Math.PI;
  const polygonK = (0.5 / 2) * (4 + 1 / reverseFraction);
  const rectangleK = (0.5 / 2) * (4 + 2);
  expect(Number.isFinite(polygonK)).toBe(true);
  expect(polygonK).toBeCloseTo(2.5, 12);
  expect(rectangleK).toBe(1.5);
  expect(polygonK - rectangleK).toBeCloseTo(1, 12);
});

it('S3 Ripley precision prose limits exact public pairCounts to uint32 capacity', () => {
  const source = readFileSync(
    new URL('../../../src/dataframe-analysis/pair-statistics/gpu-ripley.ts', import.meta.url),
    'utf8'
  );
  const paragraph = source.slice(
    source.indexOf(' * Determinism and precision:'),
    source.indexOf(' * Monte Carlo CSR')
  );
  expect(paragraph).toContain('`pairCounts` is exact only up to `0xffffffff`');
  expect(paragraph).toContain('saturates');
  // 65537*65536 is one above uint32 capacity plus 65536; internal counts remain 64-bit.
  expect(65537 * 65536).toBe(4295032832);
});
