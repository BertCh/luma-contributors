// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  getConnectionArrowLimits,
  getConnectionArrowSize,
  getConnectionElevationScale,
  getConnectionLimitReason
} from '../../../examples/deck/kepler-compute/src/resident-arc-line-plan';

const columns = {lat0: {value: 'a'}, lng0: {value: 'b'}, lat1: {value: 'c'}, lng1: {value: 'd'}};

describe('arc and line limits', () => {
  test('plain point connections are supported', () => {
    expect(getConnectionLimitReason('arc', {columnMode: 'points', columns, visConfig: {}})).toBe(
      null
    );
    expect(getConnectionLimitReason('line', {columns, visConfig: {}})).toBe(null);
  });

  test('every unsupported mode and altitude gives a specific reason', () => {
    expect(
      getConnectionLimitReason('arc', {columnMode: 'neighbors', columns, visConfig: {}})
    ).toMatch(/neighbor/);
    expect(
      getConnectionLimitReason('line', {columnMode: 'geoarrow', columns, visConfig: {}})
    ).toMatch(/GeoArrow/);
    expect(
      getConnectionLimitReason('line', {columns: {...columns, alt0: {value: 'z'}}, visConfig: {}})
    ).toMatch(/Line altitude/);
    expect(
      getConnectionLimitReason(
        'line',
        {columns: {...columns, alt0: {value: 'z'}}, visConfig: {}},
        {alt0: 'y'}
      )
    ).toMatch(/Line altitude/);
    expect(
      getConnectionLimitReason('arc', {columns: {...columns, alt: {value: 'z'}}, visConfig: {}})
    ).toMatch(/no altitude/);
  });

  test('source arrows are reported and unknown shapes fall back', () => {
    expect(
      getConnectionArrowLimits({
        columns,
        visConfig: {sourceArrowEnabled: true, targetArrowShape: 'x'}
      })
    ).toHaveLength(2);
    expect(getConnectionArrowLimits({columns, visConfig: {targetArrowShape: 'stealth'}})).toEqual(
      []
    );
    expect(getConnectionArrowSize(undefined)).toBe(16);
    expect(getConnectionArrowSize(24)).toBe(24);
  });
});

describe('line altitude', () => {
  const altitudeColumns = {...columns, alt0: {value: 'z0'}, alt1: {value: 'z1'}};

  test('prepared alt0/alt1 columns render; a missing endpoint refuses', () => {
    expect(
      getConnectionLimitReason(
        'line',
        {columns: altitudeColumns, visConfig: {}},
        {alt0: 'z0', alt1: 'z1'}
      )
    ).toBe(null);
    expect(
      getConnectionLimitReason('line', {columns: altitudeColumns, visConfig: {}}, {alt0: 'z0'})
    ).toMatch(/prepare the dataset/);
    // Without configured altitude columns the prepared altitude is irrelevant.
    expect(getConnectionLimitReason('line', {columns, visConfig: {}}, {alt0: 'z0'})).toBe(null);
    expect(
      getConnectionLimitReason('arc', {columns: altitudeColumns, visConfig: {}}, {alt0: 'z0'})
    ).toMatch(/no altitude/);
  });

  test('elevation scale applies on maps and is ignored on the globe', () => {
    expect(getConnectionElevationScale(3, false)).toBe(3);
    expect(getConnectionElevationScale(3, true)).toBe(1);
    expect(getConnectionElevationScale(undefined, false)).toBe(1);
  });
});
