// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {getCaptureRowCount} from '../../src/computation-views/views/record-queries';
function readScenario(name: string) {
  return fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/gpu-spatial-analysis.${name}.json`, import.meta.url),
        'utf8'
      )
    )
  ).record;
}
test('F12 archived regression predictors declare 199 two-component tract rows', () => {
  const predictors = readScenario('spatial-regression-recipe').captures!.snapshots.find(
    capture => capture.name === 'predictors'
  )!;
  expect(predictors.components).toBe(2);
  expect(getCaptureRowCount(predictors)).toBe(199);
});
test('F12 archived pair gather retains producer and gathered candidate status', () => {
  const names = readScenario('pair-gather').captures!.snapshots.map(capture => capture.name);
  expect(names).toContain('pairCandidateOverflow');
  expect(names).toContain('candidateOverflow');
});
