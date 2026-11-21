// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPURipleyDistanceFunctions} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-distance-functions';
import {getGPURipleyDistanceFunctionsParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-distance-parameters';
import {createPairStatisticsHarness} from './pair-statistics-harness';

it('PTS-138 exact Hanisch denominator includes uncensored tails and excludes censored events and references', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of [
    // Event d=[1,1,>1.5,>1.5], b=[3,3,1.5,1.5]. Only the close pair is uncensored.
    {
      positions: [3, 3, 4, 3, 1.5, 8, 8.5, 8],
      bounds: [0, 0, 10, 10] as const,
      grid: [1, 1] as const,
      g: 1,
      f: 0
    },
    // Reference centres (2,6),(6,6),(10,6): (d,b)=(0,2),(3,6),(7,2).
    // Eroded-area ratios are 1, 144/(6*6)=4, censored. F(1)=1/(1+4)=0.2.
    {positions: [2, 6, 3, 6], bounds: [0, 0, 12, 12] as const, grid: [3, 1] as const, g: 1, f: 0.2}
  ]) {
    const harness = createPairStatisticsHarness(device, {
      positions: Float32Array.from(fixture.positions),
      parameterLength: 8,
      outputs: {g: {format: 'float32', length: 1}, f: {format: 'float32', length: 1}},
      createContributor: views =>
        new GPURipleyDistanceFunctions({
          positions: views.positions,
          parameters: views.parameters,
          gridSize: [12, 12],
          radiusCount: 1,
          referenceGrid: fixture.grid,
          hanischSearch: 'exact',
          output: {g: views.outputs.g as never, f: views.outputs.f as never}
        })
    });
    try {
      const result = await harness.run(
        getGPURipleyDistanceFunctionsParameterValues({
          bounds: fixture.bounds,
          maximumDistance: 1,
          edgeCorrection: 'hanisch'
        })
      );
      for (const name of ['g', 'f'] as const) {
        expect(result[name]).toHaveLength(1);
        expect(Number.isFinite(result[name][0])).toBe(true);
        expect(result[name][0]).toBeCloseTo(fixture[name], 5);
      }
    } finally {
      harness.destroy();
    }
  }
});

it('PTS-140 pins G F and J to an explicit censored product-limit table', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Window [0,20]^2. Events: (d,b)=(1,4),(1,4),(3,8),(3,5),(>3,1).
  // G risk table (r,N,D,C): (1,5,2,1),(2,2,0,0),(3,2,2,0).
  // Reference centres (5,5),(15,5),(5,15),(15,15): d=1,7,5,3; b=5 each.
  // F risk table: (1,4,1,0),(2,3,0,0),(3,3,1,0).
  // Hanisch area ratios at d=1,3: 100/81,100/49. Capped F assigns the
  // latter weight to each of its two unresolved references. All numbers below follow
  // these tables, independently of the local CPU oracle.
  for (const fixture of [
    {mode: 'none' as const, g: [0.4, 0.4, 0.8], f: [0.25, 0.25, 0.5], j: [0.8, 0.8, 0.4]},
    {mode: 'kaplan-meier' as const, g: [0.4, 0.4, 1], f: [0.25, 0.25, 0.5], j: [0.8, 0.8, 0]},
    {
      mode: 'hanisch' as const,
      g: [49 / 130, 49 / 130, 1],
      f: [49 / 292, 49 / 292, 130 / 292],
      j: [81 / 130 / (243 / 292), 81 / 130 / (243 / 292), 0]
    }
  ]) {
    const harness = createPairStatisticsHarness(device, {
      positions: Float32Array.from([4, 4, 5, 4, 12, 12, 15, 12, 1, 18]),
      parameterLength: 8,
      outputs: {
        g: {format: 'float32', length: 3},
        f: {format: 'float32', length: 3},
        j: {format: 'float32', length: 3}
      },
      createContributor: views =>
        new GPURipleyDistanceFunctions({
          positions: views.positions,
          parameters: views.parameters,
          gridSize: [8, 8],
          radiusCount: 3,
          referenceGrid: [2, 2],
          output: {
            g: views.outputs.g as never,
            f: views.outputs.f as never,
            j: views.outputs.j as never
          }
        })
    });
    try {
      const result = await harness.run(
        getGPURipleyDistanceFunctionsParameterValues({
          bounds: [0, 0, 20, 20],
          maximumDistance: 3,
          edgeCorrection: fixture.mode
        })
      );
      for (const name of ['g', 'f', 'j'] as const) {
        expect(result[name]).toHaveLength(3);
        for (let bin = 0; bin < 3; bin++) {
          expect(Number.isFinite(result[name][bin])).toBe(true);
          expect(result[name][bin]).toBeCloseTo(fixture[name][bin], 5);
        }
      }
    } finally {
      harness.destroy();
    }
  }
});
