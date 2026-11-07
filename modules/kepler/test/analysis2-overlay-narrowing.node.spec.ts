// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {planCartogramOverlay} from '../../../examples/deck/kepler-compute/src/analysis2-cartogram-overlay';
import {
  describeRateSourceExclusions,
  describeWideNumericNarrowing
} from '../../../examples/deck/kepler-compute/src/analysis2-narrowing-summary';

test('overlay decision: mercator, declared plane, and table-only reasons', () => {
  const bounds = [0, 0, 1000, 1000] as const;
  expect(planCartogramOverlay('EPSG:3857', bounds).kind).toBe('web-mercator');
  expect(planCartogramOverlay('EPSG:32618', bounds).kind).toBe('table-only');
  const declaration = {metersPerUnit: 1, planeOrigin: [0, 0], anchor: [-74, 40]} as const;
  expect(planCartogramOverlay('EPSG:32618', bounds, declaration).kind).toBe('declared-plane');
  expect(planCartogramOverlay('EPSG:32618', [0, 0, 5e6, 5e6], declaration).kind).toBe('table-only');
});

test('wide column narrowing and rate exclusion summaries read Arrow tables', () => {
  const table = arrow.tableFromArrays({
    wide: new Float64Array([1, 2.1, 3]),
    big: new BigInt64Array([1n, 16777217n, 3n]),
    events: [1, null, -2],
    people: [10, 5, 0]
  });
  const lines = describeWideNumericNarrowing(table);
  expect(lines.some(line => line.includes('Narrowing wide') && line.includes('rounded'))).toBe(
    true
  );
  expect(lines.some(line => line.includes('Narrowing big') && line.includes('rounded in 1'))).toBe(
    true
  );
  expect(describeRateSourceExclusions(table, 'events', 'people')).toBe(
    'Excluded rows: 1 nullEvent, 1 negativeEvent.'
  );
});

test('narrowing and rate summaries scan every row across chunks, with nulls', () => {
  const rowCount = 250000;
  const wide = new Float64Array(rowCount).fill(1.5);
  // The only inexact value lies past the former 200,000-row prefix, in a second chunk.
  wide[rowCount - 1] = 0.1;
  const first = arrow.tableFromArrays({wide: wide.subarray(0, 200000)});
  const second = arrow.tableFromArrays({wide: wide.subarray(200000)});
  const table = new arrow.Table([...first.batches, ...second.batches]);
  const lines = describeWideNumericNarrowing(table);
  expect(lines).toHaveLength(1);
  expect(lines[0]).toContain(`rounded in 1 of ${rowCount} rows`);
  expect(lines[0]).not.toContain('first');
  const nullable = arrow.tableFromArrays({wide: [1.5, null, 0.1]});
  expect(describeWideNumericNarrowing(nullable)[0]).toContain('rounded in 1 of 3 rows');
  const events = new Float64Array(rowCount).fill(1);
  const people = new Float64Array(rowCount).fill(10);
  people[rowCount - 1] = 0;
  const rates = arrow.tableFromArrays({events, people});
  expect(describeRateSourceExclusions(rates, 'events', 'people')).toBe(
    'Excluded rows: 1 zeroPopulation.'
  );
});
