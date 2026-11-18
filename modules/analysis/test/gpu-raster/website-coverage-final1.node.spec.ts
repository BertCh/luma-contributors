// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readZoneCoverage} from '../../../../website/src/gpu/runs/entries/raster-bands-indices/support/zones';

it('F12 distinguishes pending, incomplete and valid empty coverage without painting false support', () => {
  const target = Float32Array.of(99);
  const coverage = {
    zoneIds: new Uint32Array(),
    cellIds: new Uint32Array(),
    fractions: new Float32Array(),
    count: Uint32Array.of(0),
    requiredCount: Uint32Array.of(0),
    overflow: Uint32Array.of(0),
    candidateOverflow: Uint32Array.of(0),
    zoneValidity: Uint32Array.of(0)
  };
  expect(readZoneCoverage(target, coverage, 0).status).toBe('incomplete');
  expect(target[0]).toBeNaN();
  coverage.zoneValidity[0] = 1;
  expect(readZoneCoverage(target, coverage, 0)).toEqual({
    status: 'complete',
    cells: 0,
    partial: 0,
    area: 0
  });
  expect(readZoneCoverage(target, {...coverage, count: null}, 0).status).toBe('pending');
});

it('F12 incomplete tuple or candidate evidence clears a previous positive highlight', () => {
  const target = Float32Array.of(99);
  const coverage = {
    zoneIds: Uint32Array.of(0),
    cellIds: Uint32Array.of(0),
    fractions: Float32Array.of(1),
    count: Uint32Array.of(1),
    requiredCount: Uint32Array.of(2),
    overflow: Uint32Array.of(0),
    candidateOverflow: Uint32Array.of(0),
    zoneValidity: Uint32Array.of(1)
  };
  expect(readZoneCoverage(target, coverage, 0).status).toBe('incomplete');
  expect(target[0]).toBeNaN();
  coverage.requiredCount[0] = 1;
  coverage.candidateOverflow[0] = 1;
  expect(readZoneCoverage(target, coverage, 0).status).toBe('incomplete');
  expect(target[0]).toBeNaN();
  coverage.candidateOverflow[0] = 0;
  expect(readZoneCoverage(target, coverage, 0).area).toBe(1);
  expect(target[0]).toBe(1);
});
