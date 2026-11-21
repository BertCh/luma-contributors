// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {readFileSync} from 'node:fs';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {getStatisticsCaptureFailures} from './statistics-capture-expectations';

test.each([
  [
    'GPUSpatialPanelErrorMaximumLikelihood',
    'spatial-panel-error-maximum-likelihood',
    'diagnosticStatus',
    [0]
  ],
  ['GPUProbitRegression', 'probit-regression', 'diagnosticStatus', [0]],
  ['GPUSeeminglyUnrelatedRegression', 'seemingly-unrelated-regression', 'spatialStatus', [0, 0]],
  [
    'GPUSpatialPanelLagMaximumLikelihood',
    'spatial-panel-lag-maximum-likelihood',
    'hausmanStatus',
    [2]
  ],
  ['GPUFuzzyContiguity', 'fuzzy-contiguity', 'status-requiredCount', [4]],
  ['GPUIslandAttachment', 'island-attachment', 'status-requiredCount', [94]]
] as const)('%s requires %s capture %s with its fixture words', (element, slug, name, words) => {
  const {record} = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/gpu-spatial-analysis.${slug}.json`, import.meta.url),
        'utf8'
      )
    )
  );
  const snapshots = record.captures!.snapshots;
  const capture = snapshots.find(snapshot => snapshot.name === name)!;
  expect(Array.from(capture.values)).toEqual(words);
  expect(getStatisticsCaptureFailures(element, snapshots)).toEqual([]);
  expect(
    getStatisticsCaptureFailures(
      element,
      snapshots.filter(snapshot => snapshot.name !== name)
    ).join(' ')
  ).toContain(name);
  for (const replacement of [
    {...capture, name: 'renamed'},
    {...capture, values: words.map(word => word + 1)}
  ]) {
    expect(
      getStatisticsCaptureFailures(
        element,
        snapshots.map(snapshot => (snapshot === capture ? replacement : snapshot))
      ).join(' ')
    ).toContain(name);
  }
});

test('missing, renamed and nonzero required statistics statuses fail', () => {
  const element = 'GPUSpatialCrossValidatedOLS';
  const valid = ['folds-status', 'fitStatus', 'metricStatus'].map(name => ({
    name,
    values: [0, 0, 0]
  }));
  expect(getStatisticsCaptureFailures(element, valid)).toEqual([]);
  for (const index of [0, 1, 2]) {
    const missing = valid.filter((_, row) => row !== index);
    expect(getStatisticsCaptureFailures(element, missing).join(' ')).toContain(valid[index].name);
    const renamed = valid.map((capture, row) =>
      row === index ? {...capture, name: 'renamed'} : capture
    );
    expect(getStatisticsCaptureFailures(element, renamed).join(' ')).toContain(valid[index].name);
    const failed = valid.map((capture, row) =>
      row === index ? {...capture, values: [0, 1, 0]} : capture
    );
    expect(getStatisticsCaptureFailures(element, failed).join(' ')).toContain(valid[index].name);
  }
});

test('wrong finite and NaN rank results fail the independent synthetic oracle', () => {
  const values = [4, 2, 0, 1, 3, 5, 6, 7, 7, 4, 2, 0, 1, 3, 5, 6, 7, 6, 4, 2, 0, 1, 3, 5];
  expect(getStatisticsCaptureFailures('GPURankMarkov', [{name: 'ranks', values}])).toEqual([]);
  for (const changed of [3, NaN]) {
    expect(
      getStatisticsCaptureFailures('GPURankMarkov', [
        {name: 'ranks', values: [changed, ...values.slice(1)]}
      ])
    ).toEqual(['GPURankMarkov: ranks differ from the independently sorted fixture']);
  }
});
