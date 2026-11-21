// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPURankMarkovOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation';
import {CPURankMarkovOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  AUDIT_RANK_CASES,
  AUDIT_RANK_EMPTY,
  AUDIT_RANK_MASK,
  AUDIT_RANK_VALUES
} from './audit-rank-markov-fixture';

it('audits pairwise, sorted and CPU rank operation outputs against independent SciPy literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {mode, tieOrder, expected} of AUDIT_RANK_CASES) {
    for (const affine of [false, true]) {
      for (const rankingMethod of ['pairwise', 'sorted'] as const) {
        // The GPU operation lowers to the contributor, exercising both rank shader routes.
        for (const Operation of [GPURankMarkovOperation, CPURankMarkovOperation]) {
          const actual = await runOperationGraph(device, {
            inputs: {
              values: {
                format: 'float32',
                values: Float32Array.from(AUDIT_RANK_VALUES, value =>
                  affine ? 2 * value + 3 : value
                )
              },
              mask: {format: 'uint32', values: Uint32Array.from(AUDIT_RANK_MASK)}
            },
            operations: [
              new Operation({
                values: 'values',
                mask: 'mask',
                rowCount: 8,
                periodCount: 4,
                mode,
                tieOrder,
                rankingMethod,
                output: {
                  ranks: 'ranks',
                  counts: 'counts',
                  probabilities: 'probabilities',
                  rowTotals: 'rowTotals',
                  ignored: 'ignored',
                  mobility: 'mobility',
                  steadyStates: 'steadyStates',
                  stateSparsity: 'stateSparsity',
                  status: {converged: 'converged'}
                }
              })
            ],
            outputs: Object.keys(expected)
          });
          for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
            // Only f32 probabilities/means/GTH need tolerance; integer outputs stay exact.
            expectNumericArrayClose(actual[port], expected[port], {
              absoluteTolerance: actual[port] instanceof Float32Array ? 2e-6 : 0,
              label: `${mode}, ${tieOrder}, ${rankingMethod}, ${port}`
            });
          }
        }
      }
    }
  }
});

it('audits empty-panel GPU rank outputs rather than merely accepting matching twins', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const rankingMethod of ['pairwise', 'sorted'] as const) {
    const actual = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.from(AUDIT_RANK_VALUES)},
        mask: {format: 'uint32', values: new Uint32Array(8)}
      },
      operations: [
        new GPURankMarkovOperation({
          values: 'values',
          mask: 'mask',
          rowCount: 8,
          periodCount: 4,
          rankingMethod,
          output: {
            ranks: 'ranks',
            counts: 'counts',
            probabilities: 'probabilities',
            rowTotals: 'rowTotals',
            ignored: 'ignored',
            mobility: 'mobility',
            steadyStates: 'steadyStates',
            stateSparsity: 'stateSparsity',
            status: {converged: 'converged'}
          }
        })
      ],
      outputs: Object.keys(AUDIT_RANK_EMPTY)
    });
    for (const port of Object.keys(AUDIT_RANK_EMPTY) as (keyof typeof AUDIT_RANK_EMPTY)[]) {
      expectNumericArrayClose(actual[port], AUDIT_RANK_EMPTY[port], {
        label: `${rankingMethod}, ${port}`
      });
    }
  }
});
