// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {installResidentSynchronousGuards} from '../../../examples/deck/kepler-compute/src/resident-synchronous-guards';

test('resident synchronous guards refuse row mutation and report it', () => {
  const reports: string[] = [];
  const table = {appendRows: () => 'ok', getValue: () => 1} as Record<string, unknown>;
  installResidentSynchronousGuards(table, report => reports.push(report.method));
  expect(() => (table.appendRows as () => void)()).toThrow();
  expect(reports).toEqual(['appendRows']);
  expect((table.getValue as () => number)()).toBe(1);
  expect(table.upsertRows).toBeUndefined();
});
