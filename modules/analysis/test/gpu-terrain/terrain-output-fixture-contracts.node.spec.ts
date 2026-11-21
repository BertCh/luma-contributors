// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getTerrainOperationOracles} from './terrain-output-operation-fixtures';

it('typed peak and solar fixture selections retain their exact CPU and GPU terminal sets', () => {
  const fixtures = getTerrainOperationOracles().filter(
    fixture =>
      fixture.name.startsWith('profile-parabolas/') ||
      fixture.name.startsWith('solar-NOAA-reference')
  );
  // Five peak selections and four solar selections are declared explicitly by the fixture builder.
  expect(fixtures.map(fixture => fixture.name)).toEqual([
    'profile-parabolas/all',
    'profile-parabolas/prominence',
    'profile-parabolas/refinedValue',
    'profile-parabolas/peakMask',
    'profile-parabolas/peaks',
    'solar-NOAA-reference',
    'solar-NOAA-reference/only-azimuth',
    'solar-NOAA-reference/only-altitude',
    'solar-NOAA-reference/only-daylight'
  ]);
  for (const fixture of fixtures) {
    // The expected port names belong to the independent parabola/NOAA records, not either operation.
    const expected = Object.keys(fixture.expected).sort();
    expect(expected.length).toBeGreaterThan(0);
    for (const [backend, operation] of [
      ['GPU', fixture.gpu()],
      ['CPU', fixture.cpu()]
    ] as const) {
      const {outputs} = operation.contract;
      if (!outputs || typeof outputs === 'function') {
        throw new Error(`${fixture.name} ${backend} must declare its outputs directly`);
      }
      expect(Object.keys(outputs).sort(), `${fixture.name} ${backend}`).toEqual(expected);
    }
  }
});
