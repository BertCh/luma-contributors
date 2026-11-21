// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getGPUMapMatchingParameterValues} from '../../../src/gpu-network/map-matching/gpu-map-matching';

it('rejects invalid float32 calibration at the setting that loses its meaning', () => {
  const valid = {sigma: 1, beta: 1, searchRadius: 1, routeFactor: 3, routeSlack: 0};
  for (const name of ['sigma', 'beta', 'searchRadius', 'routeFactor', 'routeSlack'] as const) {
    for (const value of [NaN, Infinity, 1e40, -1])
      expect(() => getGPUMapMatchingParameterValues({...valid, [name]: value})).toThrow(
        new RegExp(name)
      );
  }
  for (const name of ['sigma', 'beta'] as const)
    for (const value of [0, -0, 1e-50])
      expect(() => getGPUMapMatchingParameterValues({...valid, [name]: value})).toThrow(
        new RegExp(name)
      );
  expect(
    Array.from(
      getGPUMapMatchingParameterValues({
        sigma: 1,
        beta: 2,
        searchRadius: 0,
        routeFactor: 0,
        routeSlack: 0
      })
    )
  ).toEqual([1, 2, 0, 0, 0, 0, 0, 0]);
});

it('captures each numeric setting once before validation and packing', () => {
  let reads = 0;
  const packed = getGPUMapMatchingParameterValues({
    get sigma() {
      reads++;
      return 1;
    },
    beta: 2,
    searchRadius: 3
  });
  expect(reads).toBe(1);
  expect(Array.from(packed)).toEqual([1, 2, 3, 3, 50, 0, 0, 0]);
});
