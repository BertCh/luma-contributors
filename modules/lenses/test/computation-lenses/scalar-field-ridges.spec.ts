// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice, submitGraph} from '@luma-contributors/analysis/testing';
import {
  buildScalarFieldRidgesScenario,
  run
} from '../../src/computation-lenses/scenarios/gpu-raster/scalar-field-ridges.lens';
import {createCoverageTrancheOneHarness} from './raster-terrain/coverage-tranche-one-harness';

it('captures the hand-derived quadratic ridge, half-cell tie, borders and missing stencil', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createCoverageTrancheOneHarness(device, 'scalar-field-ridges', 49);
  try {
    await buildScalarFieldRidgesScenario(harness.session);
    harness.upload();
    const compiled = harness.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // f = 20 - (column - 2.5)^2: H = diag(-2, 0), normal = (1, 0).
      // Column 2 has Newton offset +0.5 (excluded); column 3 has -0.5 (included).
      // Cell 40 is (5, 5), so its interior stencil neighbors (4..5, 4..5) are invalid.
      const expectedValidity = Array.from({length: 49}, (_, index) => {
        const row = Math.floor(index / 7);
        const column = index % 7;
        return Number(row > 0 && row < 6 && column > 0 && column < 6 && !(row >= 4 && column >= 4));
      });
      expect(await harness.read(compiled, 'ridgeMask')).toEqual(
        Array.from({length: 49}, (_, index) => Number(index % 7 === 3 && index > 6 && index < 42))
      );
      expect(await harness.read(compiled, 'ridgeValidity')).toEqual(expectedValidity);
      const strength = await harness.read(compiled, 'ridgeStrength');
      for (const [index, valid] of expectedValidity.entries())
        expect(strength[index]).toBeCloseTo(valid * 2, 6);
    } finally {
      compiled.destroy();
    }
  } finally {
    harness.destroy();
  }
});

it('validates the executable ridge record and its captured sidecars', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await run(device);
  expect(result.captures?.runId).toBe(result.runId);
  expect(result.captures?.snapshots.map(capture => capture.name)).toEqual([
    'scalarFieldValues',
    'scalarFieldValidity',
    'ridgeMask',
    'ridgeValidity',
    'ridgeStrength'
  ]);
  expect(
    result.refs.formulas.some(
      formula => formula.id === 'gpu-raster.scalar-field-ridges.height-ridge'
    )
  ).toBe(true);
  expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
});
