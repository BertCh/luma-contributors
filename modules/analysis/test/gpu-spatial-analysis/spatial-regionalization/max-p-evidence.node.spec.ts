// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {createMaxPRefinementInput} from './regionalization-operation-test-utils';
import {decodeGPUMaxPRegionsEvidence} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {createPartitionInput} from './partition-budget-fixtures';

it('distinguishes an exhausted growth budget, settled construction and disabled improvement', () => {
  const short = computeMaxPRegionsOnCPU(createPartitionInput(), {maximumGrowthSteps: 2});
  expect(Array.from(short.evidence)).toEqual([1, 0, 0, 0, 0]);
  // Singleton regions already meet the zero threshold: construction settles without a move.
  const settled = computeMaxPRegionsOnCPU(createPartitionInput([[1], [0]], [0, 1], [1, 1], 0), {
    maximumGrowthSteps: 20,
    maximumIterations: 4
  });
  expect(Array.from(settled.evidence)).toEqual([0, 1, 1, 1, 0]);
});

it('carries the caller ID through shared CPU shape checks', () => {
  expect(() =>
    computeMaxPRegionsOnCPU(
      {...createPartitionInput(), values: new Float32Array(0)},
      {id: 'custom-max-p'}
    )
  ).toThrow(/^custom-max-p/);
});

it('reports improvement exhaustion without claiming convergence and decodes offset reads', () => {
  const result = computeMaxPRegionsOnCPU(createMaxPRefinementInput(4), {
    standardize: false,
    startCount: 1,
    maximumIterations: 1
  });
  // A successful first round moves row 4; the required move of row 0 still needs a second round.
  expect(Array.from(result.labels)).toEqual([1, 0, 0, 1, 1, 2, 1, 2, 2]);
  expect(Array.from(result.evidence.slice(2))).toEqual([1, 0, 1]);
  const bytes = new Uint8Array(24);
  bytes.set(new Uint8Array(result.evidence.buffer), 4);
  expect(decodeGPUMaxPRegionsEvidence(bytes.subarray(4))).toMatchObject({
    improvementEnabled: true,
    improvementConverged: false,
    iterationLimit: true
  });
  expect(() => decodeGPUMaxPRegionsEvidence(new Uint8Array(19))).toThrow(/whole/);
});

it('suppresses evidence on either adjacency loss and recovers when cleared', () => {
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const flag = Uint32Array.of(0);
    const input = {...createPartitionInput(), status: {[name]: flag}};
    for (const value of [0, 1, 0]) {
      flag[0] = value;
      const result = computeMaxPRegionsOnCPU(input, {maximumGrowthSteps: 2});
      expect(Array.from(result.evidence)).toEqual(value ? [0, 0, 0, 0, 0] : [1, 0, 0, 0, 0]);
    }
  }
});
