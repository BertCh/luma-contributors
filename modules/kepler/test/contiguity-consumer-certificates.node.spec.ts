// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, test} from 'vitest';
import {decodeGPUAnalysisContiguityStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';

test.each([
  [
    'modules/kepler/src/profile/dataset-profile-gpu.ts',
    'const result = decodeGPUAnalysisContiguityStatistics(',
    {
      countValues: Uint32Array.of(2, 0, 1, 0, 1, 1),
      medianValues: Float32Array.of(1),
      validityValues: Uint32Array.of(1),
      overflowValues: Uint32Array.of(0),
      candidateValues: Uint32Array.of(0)
    }
  ],
  [
    'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/analysis-profile-run.ts',
    'const statistics = decodeGPUAnalysisContiguityStatistics(',
    {
      counts: Uint32Array.of(2, 0, 1, 0, 1, 1),
      median: Float32Array.of(1),
      validity: Uint32Array.of(1),
      weightsOverflow: Uint32Array.of(0),
      candidateOverflow: Uint32Array.of(0)
    }
  ]
] as const)('%s preserves producer-shaped contiguity readbacks and recovery', (path, marker, reads) => {
  const source = readFileSync(path, 'utf8');
  const start = source.indexOf(marker) + marker.length;
  const end = source.indexOf(');', start);
  const expression = transpileModule(`const input = ${source.slice(start, end)};`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const getInput = new Function('environment', `with(environment) { ${expression} return input; }`);
  const check = (construct: typeof getInput) => {
    const sourceReads: Record<string, Uint32Array | Float32Array> = reads;
    const finalLoss = sourceReads['overflowValues'] ?? sourceReads['weightsOverflow'];
    const candidateLoss = sourceReads['candidateValues'] ?? sourceReads['candidateOverflow'];
    for (const [overflow, candidateOverflow] of [
      [0, 0],
      [0, 1],
      [1, 0],
      [0, 0]
    ]) {
      finalLoss[0] = overflow;
      candidateLoss[0] = candidateOverflow;
      const input = construct({...sourceReads, values: (name: string) => sourceReads[name]});
      expect(input.weights?.status).toBeDefined();
      const result = decodeGPUAnalysisContiguityStatistics(input);
      if (overflow || candidateOverflow)
        expect(result).toEqual({
          exact: false,
          overflow: Boolean(overflow),
          candidateOverflow: Boolean(candidateOverflow)
        });
      else expect(result).toMatchObject({exact: true, slots: 2, islands: 1, median: 1});
    }
  };
  check(getInput);
  // A production expression that reverses the named producer destinations must fail this test.
  expect(() =>
    check((environment: object) => {
      const input = getInput(environment);
      [input.weights.status.overflow, input.weights.status.candidateOverflow] = [
        input.weights.status.candidateOverflow,
        input.weights.status.overflow
      ];
      return input;
    })
  ).toThrow();
});
