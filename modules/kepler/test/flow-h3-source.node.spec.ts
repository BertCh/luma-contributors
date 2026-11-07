// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {cellToLatLng, latLngToCell} from 'h3-js';
import {
  FLOW_H3_COLUMNS,
  makeFlowH3Table
} from '../../../examples/deck/kepler-compute/src/flow-h3-source';

test('H3 endpoints become cell-center columns and invalid cells become NaN', () => {
  const source = latLngToCell(40.7, -74, 7);
  const target = latLngToCell(40.8, -73.9, 7);
  const table = arrow.tableFromArrays({
    origin: [source, 'invalid', source],
    destination: [target, target, null],
    count: new Float64Array([1, 2, 3])
  });
  const result = makeFlowH3Table(table, 'origin', 'destination');
  if ('reason' in result) throw new Error(result.reason);
  const [latitude, longitude] = cellToLatLng(source);
  expect(result.table.getChild(FLOW_H3_COLUMNS.sourceLongitude)!.get(0)).toBeCloseTo(longitude, 6);
  expect(result.table.getChild(FLOW_H3_COLUMNS.sourceLatitude)!.get(0)).toBeCloseTo(latitude, 6);
  expect(Number.isNaN(result.table.getChild(FLOW_H3_COLUMNS.sourceLongitude)!.get(1))).toBe(true);
  expect(Number.isNaN(result.table.getChild(FLOW_H3_COLUMNS.targetLatitude)!.get(2))).toBe(true);
  expect(result.invalidCount).toBe(2);
  expect(result.table.getChild('count')!.get(2)).toBe(3);
  expect('reason' in makeFlowH3Table(table, 'origin', 'missing')).toBe(true);
});
