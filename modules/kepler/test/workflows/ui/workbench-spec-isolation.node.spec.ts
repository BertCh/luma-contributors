// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';

// The host and pure-data suites also run this assertion in afterAll, so running this file
// before either suite cannot let later contamination pass unnoticed.
test('workbench specs leave no deck.gl 9 registration in a shared Node worker', () => {
  const version = (globalThis as typeof globalThis & {deck?: {VERSION?: unknown}}).deck?.VERSION;
  expect(
    String(version ?? ''),
    'Workbench specs must not load deck.gl 9 into the shared worker; mock Kepler and deck runtimes.'
  ).not.toMatch(/^9\./);
});
