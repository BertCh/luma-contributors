// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectHawkesDensityOutputs} from './hawkes-density-assertions';
import {TWO_EVENT_DENSITY} from './two-event-density-fixture';

for (const port of ['estimates', 'backgroundProbability', 'iterationCount'] as const) {
  it(`S2 rejects missing and extra ${port} words`, () => {
    const actual = Object.fromEntries(
      Object.entries(TWO_EVENT_DENSITY).map(([name, values]) => [name, [...values]])
    );
    const expected = actual[port];
    expectHawkesDensityOutputs(actual);
    expect(() =>
      expectHawkesDensityOutputs({...actual, [port]: [...expected, expected[0]]})
    ).toThrow();
    expect(() => expectHawkesDensityOutputs({...actual, [port]: expected.slice(0, -1)})).toThrow();
  });
}
