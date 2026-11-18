// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {
  extractLensExample,
  LENS_EXAMPLE_MAX_LINES
} from '../../src/computation-lenses/lens-example';

for (const [scenario, operators, statusConnections] of [
  [
    'areal-interpolation',
    ['GPUArealInterpolation', 'GPUSpatialLag'],
    ['status: {overflow: pairOverflow}', 'weights: {...weights, status: {overflow: pairOverflow}}']
  ],
  [
    'spatial-join-prepared',
    ['GPUSpatialJoinPrepared', 'GPUSpatialJoinCandidates', 'GPUSpatialPredicateJoin'],
    ['output: input.candidatePairs', 'output: {pairs: input.pairs}']
  ],
  [
    'nearest-feature-weights',
    ['GPUNearestFeatureJoin', 'GPUNearestFeatureWeights'],
    ['status: joinStatus', 'neighborStatus: joinStatus', 'status: {overflow, candidateOverflow}']
  ],
  [
    'pair-gather',
    ['GPUSpatialPredicateJoin', 'GPUPairGather'],
    ['output: {pairs}', 'pairs,', 'status: {overflow, candidateOverflow}']
  ]
] as const) {
  test(`${scenario} publishes at most 25 lines with its operators and completeness ports visible`, () => {
    const source = readFileSync(
      new URL(
        `../../src/computation-lenses/scenarios/gpu-spatial-analysis/${scenario}.example.ts`,
        import.meta.url
      ),
      'utf8'
    );
    const example = extractLensExample(source);
    expect(example.lineCount).toBeLessThanOrEqual(LENS_EXAMPLE_MAX_LINES);
    for (const operator of operators) expect(example.code).toContain(`new ${operator}(`);
    for (const connection of statusConnections) expect(example.code).toContain(connection);
  });
}
