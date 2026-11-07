// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runSuitability} from '@kepler.gl/actions';
import {expect, test} from 'vitest';
import {
  formatSuitabilityCategoryScoreText,
  listSuitabilityPolicyRecords,
  parseSuitabilityCategoryScoreText
} from '../../../examples/deck/kepler-compute/src/analysis4-suitability-policy-panel';
import {
  getResidentSuitabilityRequest,
  getResidentSuitabilityRequestKey
} from '../../../examples/deck/kepler-compute/src/resident-suitability-actions';

test('category score text parses, round trips and refuses non-finite scores', () => {
  const parsed = parseSuitabilityCategoryScoreText('landuse: forest=3, urban = 0.5\nsoil: clay=-1');
  expect(parsed.scores).toEqual({landuse: {forest: 3, urban: 0.5}, soil: {clay: -1}});
  expect(formatSuitabilityCategoryScoreText(parsed.scores)).toBe(
    'landuse: forest=3, urban=0.5\nsoil: clay=-1'
  );
  expect(parseSuitabilityCategoryScoreText('landuse: forest=abc').error).toBeDefined();
  expect(parseSuitabilityCategoryScoreText('landuse: forest=').error).toBeDefined();
  expect(parseSuitabilityCategoryScoreText('no separator').error).toBeDefined();
  expect(parseSuitabilityCategoryScoreText('').scores).toEqual({});
});

test('category scores reach the run request and change its key', () => {
  const operation = {
    id: 'suit',
    dataId: 'source',
    weights: {landuse: 1},
    categoryScores: {landuse: {forest: 3}}
  };
  expect(listSuitabilityPolicyRecords([operation])[0].categoryScoreText).toBe('landuse: forest=3');
  const state = {visState: {suitabilities: [operation]}};
  const request = getResidentSuitabilityRequest(state, runSuitability('suit'), () => true)!;
  expect(request.categoryScores).toEqual({landuse: {forest: 3}});
  expect(getResidentSuitabilityRequestKey(request)).not.toBe(
    getResidentSuitabilityRequestKey({...request, categoryScores: undefined})
  );
});
