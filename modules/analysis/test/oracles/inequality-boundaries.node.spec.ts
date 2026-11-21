// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import fixture from './fixtures/parity-wp6-inequality.json';

function expectInequalityBoundaries(reference: typeof fixture): void {
  const mixed = reference.cases.find(
    entry => entry.name === 'zero-ties-empty-mask-invalid-weights'
  );
  expect(mixed).toBeDefined();
  expect(mixed!.zoneCount).toBe(6);
  const boundaries = mixed!.runs.filter(entry => entry.settings.epsilon >= 1);
  expect(boundaries.map(run => run.settings.epsilon)).toEqual([1, 2]);
  for (const run of boundaries) {
    for (const values of [run.expected.mean, run.expected.atkinson, run.expected.theilL]) {
      expect(values).toHaveLength(6);
    }
    // Zone 0 has weights [1,.5,2], incomes [0,0,12]: mean=24/3.5=48/7.
    // For epsilon >= 1, replacing each zero by t>0 gives equivalent income
    // tending to zero; 1 - equivalentIncome/mean tends to 1. Theil L diverges.
    expect(run.expected.mean[0]).toBe(48 / 7);
    expect(run.expected.atkinson[0]).toBe(1);
    expect(run.expected.theilL[0]).toBeNull();
    expect(run.expected.atkinson[3]).toBeNull(); // all-zero zone
    expect(run.expected.mean[3]).toBe(0);
    expect(run.expected.atkinson[5]).toBeNull(); // empty zone
    expect(run.expected.mean[5]).toBeNull();
  }
  for (const name of ['all-zero-income', 'no-included-rows']) {
    const control = reference.cases.find(entry => entry.name === name);
    expect(control).toBeDefined();
    const zoneCount = name === 'all-zero-income' ? 3 : 2;
    expect(control!.zoneCount).toBe(zoneCount);
    expect(control!.runs.map(run => run.settings.epsilon)).toEqual([0, 0.5, 1, 2]);
    for (const run of control!.runs) {
      expect(run.expected.atkinson).toEqual(Array.from({length: zoneCount}, () => null));
      expect(run.expected.theilL).toEqual(Array.from({length: zoneCount}, () => null));
      expect(run.expected.mean).toHaveLength(zoneCount);
    }
  }
}

it('NUM-09 records the continuous Atkinson extension separately from Theil L', () => {
  expectInequalityBoundaries(fixture);
});

it('B8 rejects missing boundary settings', () => {
  const reference = structuredClone(fixture);
  const mixed = reference.cases.find(
    entry => entry.name === 'zero-ties-empty-mask-invalid-weights'
  )!;
  for (let runIndex = mixed.runs.length - 1; runIndex >= 0; runIndex--) {
    if (mixed.runs[runIndex].settings.epsilon >= 1) {
      mixed.runs.splice(runIndex, 1);
    }
  }
  expect(() => expectInequalityBoundaries(reference)).toThrow();
});

it.each([
  'all-zero-income',
  'no-included-rows'
])('B8 rejects missing or extra zone words in %s', name => {
  for (const values of [[], [null, null, null, null]]) {
    const reference = structuredClone(fixture);
    const control = reference.cases.find(entry => entry.name === name)!;
    control.runs[0].expected.atkinson = values;
    expect(() => expectInequalityBoundaries(reference)).toThrow();
  }
});
