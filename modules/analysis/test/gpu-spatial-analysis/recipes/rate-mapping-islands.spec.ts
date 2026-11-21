// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addSpatialRateRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {RecipeTestFixture} from './recipe-harness';

it.each([
  'negative-events',
  'zero-weights'
] as const)('W5-17/B1 excludes %s donors when detecting islands', async route => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'negative-donor');
  try {
    const events = fixture.input(
      'events',
      route === 'negative-events' ? Float32Array.of(6, -2) : Float32Array.of(4, 8),
      'float32',
      2
    );
    const populations = fixture.input(
      'populations',
      route === 'negative-events' ? Float32Array.of(30, 10) : Float32Array.of(10, 10),
      'float32',
      2
    );
    const weights = {
      offsets: fixture.input('offsets', Uint32Array.of(0, 1, 2), 'uint32', 3),
      neighbors: fixture.input('neighbors', Uint32Array.of(1, 0), 'uint32', 2),
      weights: fixture.input(
        'weights',
        route === 'negative-events' ? Float32Array.of(1, 1) : Float32Array.of(0, 0),
        'float32',
        2
      )
    };
    const outputs = [undefined, 'own-rate' as const].map((islandPolicy, index) => {
      const output = fixture.output(`output-${index}`, 'float32', 2);
      addSpatialRateRecipe(fixture.graph, {
        id: `rate-${index}`,
        events,
        populations,
        weights,
        islandPolicy,
        output: output.view
      });
      return output;
    });
    fixture.run();
    expect(await fixture.readFloat32(outputs[0], 2)).toEqual([NaN, NaN]);
    const ownRate = await fixture.readFloat32(outputs[1], 2);
    expect(ownRate).toHaveLength(2);
    // Only the focus is eligible: 6 / 30 = .2; row one is invalid in either policy.
    expect(Number.isFinite(ownRate[0])).toBe(true);
    expect(ownRate[0]).toBeCloseTo(route === 'negative-events' ? 0.2 : 0.4, 6);
    if (route === 'negative-events') expect(ownRate[1]).toBeNaN();
    else {
      expect(Number.isFinite(ownRate[1])).toBe(true);
      expect(ownRate[1]).toBeCloseTo(0.8, 6);
    }
  } finally {
    fixture.destroy();
  }
});
