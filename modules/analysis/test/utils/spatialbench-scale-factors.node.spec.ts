// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getSpatialBenchRowCounts} from './spatialbench-scale-factors';

it('reproduces the published SpatialBench row-count formulas', () => {
  expect(getSpatialBenchRowCounts(1)).toEqual({
    scaleFactor: 1,
    trip: 6000000,
    building: 20000,
    customer: 30000,
    driver: 500,
    vehicle: 100,
    zone: 156095
  });
  const scaleFactorTen = getSpatialBenchRowCounts(10);
  expect(scaleFactorTen.trip).toBe(60000000);
  expect(scaleFactorTen.zone).toBe(455711);
  expect(scaleFactorTen.building).toBe(Math.round(20000 * (1 + Math.log2(10))));
  expect(getSpatialBenchRowCounts(0.1).trip).toBe(600000);
  expect(() => getSpatialBenchRowCounts(0)).toThrow();
});
