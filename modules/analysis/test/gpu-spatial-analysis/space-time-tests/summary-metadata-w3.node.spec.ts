// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUAnalysisSmallestPseudoPValue} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-interpretation';
import {
  decodeGPUMantelTestSummary,
  GPU_MANTEL_TEST_SUMMARY,
  decodeGPUSpaceTimeInferenceMetadata,
  getGPUSpaceTimeInferenceMetadata
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';

it('W3-02 identifies the doubled minimum tail and its 19-draw floor', () => {
  const metadata = getGPUSpaceTimeInferenceMetadata(
    'mantel',
    {seed: 0, permutationCount: 19},
    19,
    undefined,
    'pseudoPTwoSided'
  );
  expect(metadata).toMatchObject({
    alternative: 'two-sided',
    pValueConstruction: 'doubled-minimum-tail'
  });
  // Zero exceedances: two times (0+1)/(19+1) = .10; a maximum-statistic upper tail is .05.
  expect(getGPUAnalysisSmallestPseudoPValue(19, metadata.pValueConstruction)).toBe(0.1);
  expect(getGPUAnalysisSmallestPseudoPValue(19, 'upper-tail-count')).toBe(0.05);
  expect(
    getGPUSpaceTimeInferenceMetadata('knox', {seed: 0, permutationCount: 19}, 19).pValueConstruction
  ).toBe('upper-tail-count');
});

it('NUM-03 decodes Mantel words and byte subviews and rejects truncated layouts', () => {
  const words = Float32Array.from({length: 12}, (_, index) => index + 0.5);
  const bytes = new Uint8Array(51);
  bytes.set(new Uint8Array(words.buffer), 3);
  const expected = {
    observedCorrelation: 0.5,
    pairCount: 1.5,
    timeClosePairs: 2.5,
    expected: 3.5,
    permutationMean: 4.5,
    permutationVariance: 5.5,
    greaterCount: 6.5,
    lesserCount: 7.5,
    pseudoPGreater: 8.5,
    pseudoPLesser: 9.5,
    pseudoPTwoSided: 10.5,
    zScore: 11.5
  };
  expect(decodeGPUMantelTestSummary(bytes.subarray(3))).toEqual(expected);
  expect(decodeGPUMantelTestSummary(words)).toEqual(expected);
  expect(GPU_MANTEL_TEST_SUMMARY).not.toHaveProperty('observed');
  expect(GPU_MANTEL_TEST_SUMMARY.observedCorrelation).toBe(0);
  expect(decodeGPUMantelTestSummary(words.buffer).observedCorrelation).toBe(0.5);
  for (const incomplete of [new Float32Array(11), new Uint8Array(47), new ArrayBuffer(46)])
    expect(() => decodeGPUMantelTestSummary(incomplete)).toThrow();
});

it('NUM-03 decodes the usable count below a fixed Mantel request', () => {
  const metadata = getGPUSpaceTimeInferenceMetadata('mantel', {seed: 0, permutationCount: 19}, 19);
  const legacyCountDefinition = {permutationCountDefinition: 'usable-correlations'};
  expect(() =>
    decodeGPUSpaceTimeInferenceMetadata({...metadata, ...legacyCountDefinition}, 19, 16)
  ).toThrow(/permutationCountDefinition was removed.*inferenceCountDefinition/);
  expect(decodeGPUSpaceTimeInferenceMetadata(metadata, 19, 16)).toMatchObject({
    available: true,
    permutationCount: 19,
    usablePermutationCount: 16,
    inference: {permutationCount: 16}
  });
  expect(decodeGPUSpaceTimeInferenceMetadata(metadata, 0)).toMatchObject({
    available: false,
    permutationCount: 0
  });
});

it('S4 rejects unknown summary fields with the accepted names', () => {
  expect(() =>
    getGPUSpaceTimeInferenceMetadata(
      'mantel',
      {seed: 0, permutationCount: 19},
      19,
      undefined,
      'typo' as never
    )
  ).toThrow(/summaryField.*pseudoPGreater.*pseudoPLesser.*pseudoPTwoSided.*typo/);
});
