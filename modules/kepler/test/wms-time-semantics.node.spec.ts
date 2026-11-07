// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  findWmsTimeExtentText,
  resolveWmsTimeValue
} from '../../../examples/deck/kepler-compute/src/wms-time-extent';

const EXTENT = '2024-01-01T00:00:00Z/2024-01-05T00:00:00Z/P1D,2024-02-01T00:00:00Z';

describe('wms time semantics', () => {
  it('resolves current to the nearest advertised instant', () => {
    const now = Date.parse('2024-01-03T10:00:00Z');
    expect(resolveWmsTimeValue(EXTENT, 'current', {now})).toEqual({
      kind: 'current',
      time: '2024-01-03T00:00:00Z'
    });
    expect(
      resolveWmsTimeValue(EXTENT, 'current', {now: Date.parse('2030-01-01T00:00:00Z')}).time
    ).toBe('2024-02-01T00:00:00Z');
  });
  it('resolves default to the capabilities default or the latest instant', () => {
    expect(
      resolveWmsTimeValue(EXTENT, 'default', {defaultValue: '2024-01-02T00:00:00Z'}).time
    ).toBe('2024-01-02T00:00:00Z');
    expect(resolveWmsTimeValue(EXTENT, 'default').time).toBe('2024-02-01T00:00:00Z');
  });
  it('snaps off-grid instants to the nearest advertised one', () => {
    expect(resolveWmsTimeValue(EXTENT, '2024-01-02T20:00:00Z', {allowNearest: true})).toEqual({
      kind: 'nearest',
      time: '2024-01-03T00:00:00Z'
    });
    expect(resolveWmsTimeValue(EXTENT, '2024-01-02T00:00:00Z').kind).toBe('exact');
    expect(resolveWmsTimeValue(EXTENT, '2023-01-02T20:00:00Z', {allowNearest: false}).kind).toBe(
      'none'
    );
    expect(resolveWmsTimeValue(EXTENT, 'garbage').kind).toBe('none');
  });
  it('reads the 1.1.1 Extent element with default', () => {
    const xml =
      '<Layer><Name>sst</Name><Extent name="time" default="2024-01-02T00:00:00Z">2024-01-01T00:00:00Z,2024-01-02T00:00:00Z</Extent></Layer>';
    expect(findWmsTimeExtentText(xml, 'sst')?.defaultValue).toBe('2024-01-02T00:00:00Z');
  });
});
