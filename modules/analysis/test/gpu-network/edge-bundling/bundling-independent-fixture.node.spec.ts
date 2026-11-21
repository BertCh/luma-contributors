// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {bundleEdgesOracle} from './edge-bundling-oracle';
import {
  BUNDLING_INDEPENDENT_PATHS,
  BUNDLING_INDEPENDENT_SETTINGS
} from './bundling-independent-fixture';

it('the float32 bundling reference agrees with the independent full-schedule fixture', () => {
  const {paths} = bundleEdgesOracle({
    positions: Float32Array.of(100, 450, 900, 450, 100, 550, 900, 550),
    sources: [0, 2],
    targets: [1, 3],
    pointsPerEdge: 5,
    iterations: 8,
    densityResolution: 64,
    ...BUNDLING_INDEPENDENT_SETTINGS
  });
  expectIndependentPaths(paths);
});

function expectIndependentPaths(paths: Float32Array): void {
  expect(paths).toHaveLength(BUNDLING_INDEPENDENT_PATHS.length);
  expect(Array.from(paths).every(Number.isFinite)).toBe(true);
  expect(
    Math.max(
      ...Array.from(paths, (value, index) => Math.abs(value - BUNDLING_INDEPENDENT_PATHS[index]))
    )
  ).toBeLessThan(0.25);
}

it.each(
  [[], BUNDLING_INDEPENDENT_PATHS.slice(0, 18), [...BUNDLING_INDEPENDENT_PATHS, 0]].map(values => ({
    values
  }))
)('rejects an incomplete or oversized coordinate result: $values', ({values}) => {
  expect(() => expectIndependentPaths(Float32Array.from(values))).toThrow();
});
