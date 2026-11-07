// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {listJoinPolicyRecords} from '../../../examples/deck/kepler-compute/src/analysis4-join-policy-panel';
import {listSuitabilityPolicyRecords} from '../../../examples/deck/kepler-compute/src/analysis4-suitability-policy-panel';

test('join policy records skip spatial joins and default to the native behavior', () => {
  const records = listJoinPolicyRecords([
    {id: 'a'},
    {id: 'b', implementation: 'spatial'},
    {
      id: 'c',
      resultLabel: 'Joined',
      duplicateRightMatches: 'last',
      rightColumnSuffix: '_r',
      dropRightKey: true
    }
  ]);
  expect(records.map(record => record.id)).toEqual(['a', 'c']);
  expect(records[0]).toMatchObject({
    duplicateRightMatches: 'all',
    rightColumnSuffix: undefined,
    dropRightKey: false
  });
  expect(records[1]).toMatchObject({label: 'Joined', rightColumnSuffix: '_r', dropRightKey: true});
});

test('suitability policy records default to refuse', () => {
  expect(
    listSuitabilityPolicyRecords([
      {id: 'a'},
      {id: 'b', missingValues: 'renormalize', resultLabel: 'Score'},
      {id: 'c', missingValues: 'bogus'}
    ]).map(record => record.missingValues)
  ).toEqual(['refuse', 'renormalize', 'refuse']);
});
