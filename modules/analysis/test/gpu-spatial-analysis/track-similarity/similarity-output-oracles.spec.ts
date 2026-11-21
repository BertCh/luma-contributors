// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUTrackSimilarityOperation} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {poisonOperation} from '../trajectory-cleaning/trajectory-output-oracle-utils';
import {
  getSimilarityOracleInput,
  expectSimilarityOracleOutputs,
  SIMILARITY_MEASURES,
  SIMILARITY_OPTIONS
} from './similarity-output-oracles-fixture';

it('similarity operation poisoned measures match exhaustive literals in isolation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const withBaseline of [false, true])
    for (const optional of [...SIMILARITY_MEASURES, 'none'] as const) {
      const input = getSimilarityOracleInput();
      const expectedKeys = withBaseline
        ? ['maximumDistance', 'pairStatus', ...(optional === 'none' ? [] : [optional])]
        : [optional === 'none' ? 'maximumDistance' : optional];
      const operation = new GPUTrackSimilarityOperation({
        ...SIMILARITY_OPTIONS,
        linesA: {positions: 'positionsA', lineOffsets: 'offsetsA'},
        linesB: {positions: 'positionsB', lineOffsets: 'offsetsB'},
        pairA: 'pairA',
        pairB: 'pairB',
        pairCount: 2,
        ...(optional === 'frechetWithin' ? {frechetEpsilon: 'epsilon'} : {}),
        output: Object.fromEntries(expectedKeys.map(port => [port, port]))
      });
      const result = await runOperationGraph(device, {
        inputs: {
          positionsA: {format: 'float32x2', values: input.linesA.positions},
          offsetsA: {format: 'uint32', values: input.linesA.lineOffsets},
          positionsB: {format: 'float32x2', values: input.linesB.positions},
          offsetsB: {format: 'uint32', values: input.linesB.lineOffsets},
          pairA: {format: 'uint32', values: input.pairA},
          pairB: {format: 'uint32', values: input.pairB},
          ...(optional === 'frechetWithin'
            ? {epsilon: {format: 'float32' as const, values: input.frechetEpsilon}}
            : {})
        },
        operations: [poisonOperation(operation)],
        outputs: expectedKeys
      });
      expectSimilarityOracleOutputs(result, expectedKeys);
    }
});
