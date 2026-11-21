// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNeighborhoodSummaryOnCPU} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {computeContiguityWeightsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {createPairStarvedContiguityFixture} from '../spatial-weights/contiguity-loss.test-utils';

it('accepts the producer result directly and preserves candidate loss', () => {
  const fixture = createPairStarvedContiguityFixture();
  const partial = computeContiguityWeightsOnCPU(fixture.input, fixture.options);
  const complete = computeContiguityWeightsOnCPU(fixture.input, {
    ...fixture.options,
    pairCapacity: 32
  });
  // Two adjacent squares each have one neighbor; one retained pair candidate is incomplete.
  expect(Array.from(partial.status.candidateOverflow)).toEqual([1]);
  for (const producer of [
    partial,
    {...partial, status: {...partial.status}},
    {...complete, status: {...complete.status, candidateOverflow: Uint32Array.of(1)}},
    {...complete, status: {...complete.status, complete: Uint32Array.of(0)}}
  ]) {
    const result = computeNeighborhoodSummaryOnCPU(
      {...producer, values: Float32Array.of(10, 20)},
      {statistics: ['count']}
    );
    expect(Array.from(result.table)).toEqual([NaN, NaN]);
  }
  for (const name of ['candidateOverflow', 'weightsCandidateOverflow'] as const) {
    const conflictingEvidence = {
      ...complete,
      [name]: Uint32Array.of(1),
      status: {...complete.status, candidateOverflow: Uint32Array.of(0)}
    };
    expect(() =>
      computeNeighborhoodSummaryOnCPU(
        {...conflictingEvidence, values: Float32Array.of(10, 20)},
        {statistics: ['count']}
      )
    ).toThrow(`weights.${name} was removed; use weights.status.candidateOverflow`);
  }
  expect(
    Array.from(
      computeNeighborhoodSummaryOnCPU(
        {...complete, values: Float32Array.of(10, 20)},
        {statistics: ['count']}
      ).table
    )
  ).toEqual([1, 1]);
});

it('retains convergence and iteration checks after reading the shared CPU status', () => {
  const weights = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(1, 0),
    weights: Float32Array.of(1, 1),
    values: Float32Array.of(10, 20)
  };
  for (const flags of [
    {weightsConverged: Uint32Array.of(0)},
    {weightsIterationLimitReached: Uint32Array.of(1)}
  ]) {
    expect(
      Array.from(
        computeNeighborhoodSummaryOnCPU({...weights, ...flags}, {statistics: ['count']}).table
      )
    ).toEqual([NaN, NaN]);
  }
  expect(
    Array.from(
      computeNeighborhoodSummaryOnCPU(
        {
          ...weights,
          weightsConverged: Uint32Array.of(1),
          weightsIterationLimitReached: Uint32Array.of(0)
        },
        {statistics: ['count']}
      ).table
    )
  ).toEqual([1, 1]);
});
