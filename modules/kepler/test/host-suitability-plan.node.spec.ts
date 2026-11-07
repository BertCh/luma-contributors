// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  computeNativeSuitabilityScores,
  createResidentSuitabilityPlan,
  type ResidentSuitabilityOperation
} from '../../../examples/deck/kepler-compute/src/resident-suitability-plan';

const fields = [
  {name: 'id', type: 'integer'},
  {name: 'cost', type: 'real'},
  {name: 'access', type: 'real'},
  {name: 'name', type: 'string'},
  {name: 'flag', type: 'boolean'}
];
const operation: ResidentSuitabilityOperation = {
  id: 'op',
  dataId: 'data',
  weights: {cost: 0.5, access: 1},
  weightStandardization: 'normalize',
  dataStandardization: 'range',
  outputFieldName: 'score',
  resultId: 'result',
  resultLabel: 'Suitability'
};

test('plans map native weights, standardization and score naming onto the composite scorer', () => {
  const plan = createResidentSuitabilityPlan(operation, fields);
  expect(plan).toMatchObject({scaler: 'min-max', scoreScale: 1, scoreColumn: 'score'});
  expect(
    createResidentSuitabilityPlan({...operation, dataStandardization: 'zScore'}, fields).scaler
  ).toBe('z-score');
  expect(
    createResidentSuitabilityPlan({...operation, weightStandardization: 'raw'}, fields).scoreScale
  ).toBe(1.5);
  const renamed = createResidentSuitabilityPlan(
    {...operation, outputFieldName: 'cost', columns: ['id', 'cost']},
    fields
  );
  expect(renamed.includedColumns).toEqual(['id', 'cost']);
  expect(renamed.scoreColumn).toBe('cost_1');
});

test('plans reject what the resident scorer cannot reproduce exactly', () => {
  const reject = (change: Partial<ResidentSuitabilityOperation>, pattern: RegExp) =>
    expect(() => createResidentSuitabilityPlan({...operation, ...change}, fields)).toThrow(pattern);
  reject({weights: {}}, /at least one/);
  reject({weights: {flag: 1}}, /not numeric or categorical/);
  reject({weights: {missing: 1}}, /not found/);
  reject({weights: {cost: -0.1}}, /nonnegative/);
  reject({weights: {cost: 0}}, /positive/);
  // Raw data standardization is supported (scaler none); only unknown values are rejected.
  reject({dataStandardization: 'bogus' as never}, /Unknown suitability data/);
  expect(() =>
    createResidentSuitabilityPlan(
      {
        ...operation,
        weights: Object.fromEntries(Array.from({length: 129}, (_, index) => [`f${index}`, 1]))
      },
      Array.from({length: 129}, (_, index) => ({name: `f${index}`, type: 'real'}))
    )
  ).toThrow(/at most 128/);
  // More than the former 16 fields are scored (one column at a time on the GPU).
  expect(
    createResidentSuitabilityPlan(
      {
        ...operation,
        weights: Object.fromEntries(Array.from({length: 40}, (_, index) => [`f${index}`, 1]))
      },
      Array.from({length: 40}, (_, index) => ({name: `f${index}`, type: 'real'}))
    ).indicators
  ).toHaveLength(40);
});

test('weights above 1 are rescaled with a visible note', () => {
  const plan = createResidentSuitabilityPlan({...operation, weights: {cost: 2, access: 4}}, fields);
  expect(plan.weightRescale).toBe(4);
  expect(plan.indicators.map(indicator => indicator.weight)).toEqual([0.5, 1]);
  expect(plan.notes.join(' ')).toMatch(/rescaled by 1\/4/);
  expect(plan.notes.join(' ')).toMatch(/normalized scores are unchanged/);
  const raw = createResidentSuitabilityPlan(
    {...operation, weights: {cost: 2, access: 4}, weightStandardization: 'raw'},
    fields
  );
  expect(raw.scoreScale).toBe(1.5);
  expect(raw.notes.join(' ')).toMatch(/raw scores shrink/);
  expect(createResidentSuitabilityPlan(operation, fields).notes).toEqual([]);
});

test('reference scores reproduce the pinned native operation', () => {
  const cost = [0, 5, 10, null];
  const access = [10n, 20n, 30n, 40n];
  const scores = computeNativeSuitabilityScores([cost, access], [0.5, 1], {
    dataStandardization: 'range',
    weightStandardization: 'normalize'
  });
  expect(scores[0]).toBeCloseTo(0);
  expect(scores[1]).toBeCloseTo(0.5 / 3 + (1 / 3) * (2 / 3));
  expect(scores[2]).toBeCloseTo(1 / 3 + (2 / 3) * (2 / 3));
  expect(scores[3]).toBeNull();
  // Constant columns stay unchanged under native range standardization.
  expect(
    computeNativeSuitabilityScores([[3, 3]], [1], {
      dataStandardization: 'range',
      weightStandardization: 'normalize'
    })
  ).toEqual([3, 3]);
  const z = computeNativeSuitabilityScores([[1, 2, 3]], [1], {
    dataStandardization: 'zScore',
    weightStandardization: 'raw'
  });
  expect(z[0]).toBeCloseTo(-Math.sqrt(1.5));
  expect(z[1]).toBeCloseTo(0);
});
