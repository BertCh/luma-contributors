// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURankMarkovOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation';
import {
  GPURankMarkovOperation,
  type RankMarkovOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const mode of ['full-rank', 'geographic-rank'] as const)
  for (const tieOrder of ['giddy', 'row-id'] as const)
    for (const excluded of [false, true]) {
      it(`rank Markov ${mode}, ${tieOrder}, excluded=${excluded} agrees on every output`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const props: RankMarkovOperationProps = {
          values: 'values',
          mask: 'mask',
          rowCount: 4,
          periodCount: 4,
          mode,
          tieOrder,
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
        };
        const inputs = {
          values: {
            format: 'float32' as const,
            values: Float32Array.of(4, 3, 2, 1, 3, 2, 4, 1, 2, 4, 3, 1, 4, 4, 2, 1)
          },
          mask: {
            format: 'uint32' as const,
            values: excluded ? new Uint32Array(4) : Uint32Array.of(1, 1, 1, 0)
          }
        };
        const outputs = [
          'ranks',
          'counts',
          'probabilities',
          'rowTotals',
          'ignored',
          'mobility',
          'steadyStates',
          'stateSparsity',
          'converged'
        ];
        const gpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new GPURankMarkovOperation(props)]
        });
        const cpu = await runOperationGraph(device, {
          inputs,
          outputs,
          operations: [new CPURankMarkovOperation(props)]
        });
        for (const port of outputs)
          for (let index = 0; index < cpu[port].length; index++) {
            const expected = cpu[port][index];
            const actual = gpu[port][index];
            if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
            else if (cpu[port] instanceof Uint32Array || expected === 0)
              expect(actual === expected).toBe(true);
            // Relative tolerance 2e-6; positive small values cannot pass as zero.
            else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-6 * Math.abs(expected));
          }
      });
    }
