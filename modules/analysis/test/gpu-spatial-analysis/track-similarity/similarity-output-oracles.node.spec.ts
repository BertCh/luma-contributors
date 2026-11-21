// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrackSimilarityOnCPU} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {
  getSimilarityOracleInput,
  SIMILARITY_EXPECTED,
  SIMILARITY_MEASURES,
  SIMILARITY_OPTIONS
} from './similarity-output-oracles-fixture';
import {expectSimilarityOracleOutputs} from './similarity-output-oracles-fixture';

it('R7 similarity assertions fail if a requested optional port disappears', () => {
  for (const measure of SIMILARITY_MEASURES)
    expect(() => expectSimilarityOracleOutputs({}, [measure])).toThrow();
});

it('similarity twin matches exhaustive path and alignment literals on unequal tracks', () => {
  const result = computeTrackSimilarityOnCPU(getSimilarityOracleInput(), {
    ...SIMILARITY_OPTIONS,
    measures: [...SIMILARITY_MEASURES, 'maximumDistance']
  });
  for (const [port, expected] of Object.entries(SIMILARITY_EXPECTED))
    expected.forEach((value, index) => {
      const actual = result[port as keyof typeof result][index];
      expect(Number.isFinite(actual), port).toBe(true);
      // CPU output rounds once to float32. Two ulps cover this small fixture.
      expect(Math.abs(actual - value), port).toBeLessThanOrEqual(2e-6);
    });
});
