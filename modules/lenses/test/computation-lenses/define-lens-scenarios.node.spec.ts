// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  DEFAULT_LENS_RUNGS,
  createLensScenarioDescriptor
} from '../../src/computation-lenses/define-lens-scenarios';
import {runLensScenario} from '../../src/computation-lenses/run-lens-scenario';

test('defineLensScenario derives entry, audience, rungs and defaultRung from a data row', () => {
  const descriptor = createLensScenarioDescriptor({
    id: 'define-test.row-defaults',
    element: 'RowDefaults',
    title: 'Row defaults',
    load: async () => ({run: async () => ({}) as never})
  });
  expect(descriptor?.entry).toBe('define-test');
  expect(descriptor?.audience).toBe('analyst');
  expect(descriptor?.rungs).toEqual(DEFAULT_LENS_RUNGS);
  expect(descriptor?.defaultRung).toBe('result');
});

test('runLensScenario honours an aborted signal before loading', async () => {
  const controller = new AbortController();
  controller.abort();
  let loaded = false;
  await expect(
    runLensScenario(
      {
        load: async () => {
          loaded = true;
          return {run: async () => ({}) as never};
        }
      },
      {} as never,
      {signal: controller.signal}
    )
  ).rejects.toBeTruthy();
  expect(loaded).toBe(false);
});
