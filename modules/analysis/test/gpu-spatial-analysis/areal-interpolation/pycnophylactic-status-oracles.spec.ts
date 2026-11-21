// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPycnophylacticInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

// Hand oracle, independent of the shader and the TS reference:
// zones [0,0,1], totals [2,4,3] initialize to [1,1,4]; zone 2 has no cell.
// Rook mean 1: [1,5/2,1]. Restore totals: [1/4,7/4,4]. Relative change = 3/4.
// Rook mean 2: [7/4,17/8,7/4]. Restore: [13/16,19/16,4]. Change = 9/16.
// No clamp or rescale changes these positive, exactly mass-preserving results.
// Thus tolerance 0.6 first succeeds at iteration 2; tolerance 0.5 still fails at 2.
// All masses are dyadic rationals, but allow 2e-6 for float32 segment reductions.
// Totals [4,8,3] double the represented densities: step 1 [1/2,7/2,8],
// step 2 [13/8,19/8,8]. Largest absolute change 9/8 divided by zone mean 2
// is exactly 9/16. Tolerance 9/16 must stop inclusively at step 2, rejecting
// both an absolute-change comparison and a strict relative '<' comparison.
for (const testCase of [
  {budget: 0, tolerance: 0.6, status: [0, 0, 1], mass: [1, 1, 4]},
  {budget: 1, tolerance: 0.6, status: [0, 1, 1], mass: [0.25, 1.75, 4]},
  {budget: 2, tolerance: 0.5, status: [0, 2, 1], mass: [0.8125, 1.1875, 4]},
  {budget: 4, tolerance: 0.6, status: [1, 2, 1], mass: [0.8125, 1.1875, 4]},
  {budget: 2, tolerance: 9 / 16, totals: [4, 8, 3], status: [1, 2, 1], mass: [1.625, 2.375, 8]}
]) {
  it(`pycnophylactic status follows the hand-derived stopping decision (budget ${testCase.budget}, tolerance ${testCase.tolerance})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const zones = input('zones', 'uint32', Uint32Array.of(0, 0, 1));
    const mass = output('mass', 'float32', 3);
    const converged = output('converged', 'uint32', 1);
    const iterationCount = output('iterations', 'uint32', 1);
    const invalidCount = output('invalid', 'uint32', 1);
    graph.add(
      new GPUPycnophylacticInterpolation({
        grid: makeGPURasterMetadata({width: 3, height: 1}),
        zones,
        zoneCount: 3,
        totals: input('totals', 'float32', Float32Array.from(testCase.totals ?? [2, 4, 3])),
        maximumIterations: testCase.budget,
        tolerance: input('tolerance', 'float32', Float32Array.of(testCase.tolerance)),
        kernel: 'rook',
        output: {mass, status: {converged, iterationCount, invalidCount}}
      })
    );
    const compiled = graph.compile();
    const readResult = async () => ({
      status: await fixture.readRows([converged, iterationCount, invalidCount]),
      mass: await fixture.read(mass)
    });
    const expectMass = (actual: number[], expected: number[]) => {
      expect(actual).toHaveLength(expected.length);
      expected.forEach((value, row) => {
        expect(Number.isFinite(actual[row])).toBe(true);
        expect(Math.abs(actual[row] - value)).toBeLessThanOrEqual(2e-6);
      });
    };
    try {
      const result = await fixture.submitTwice(compiled, readResult);
      expect(result.status).toEqual(testCase.status);
      expectMass(result.mass, testCase.mass);

      // One cell per zone: each restore returns exactly its total, so change is 0
      // and convergence occurs at iteration 1 (unless the explicit budget is 0).
      // Reuse the compiled graph to require invalidCount 1 -> 0 and iterationCount 2 -> 1.
      fixture.writeInput(zones, Uint32Array.of(0, 2, 1));
      const restored = await fixture.submitTwice(compiled, readResult);
      expect(restored.status).toEqual(testCase.budget === 0 ? [0, 0, 0] : [1, 1, 0]);
      expectMass(restored.mass, testCase.totals ? [4, 3, 8] : [2, 3, 4]);

      fixture.writeInput(zones, Uint32Array.of(0, 0, 1));
      const repeated = await fixture.submitTwice(compiled, readResult);
      expect(repeated.status).toEqual(testCase.status);
      expectMass(repeated.mass, testCase.mass);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
