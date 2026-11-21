// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUNetworkCostTableOperation} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectOracleRows, runPoisonedOperation} from '../alternative-routes/output-oracle-harness';
import {costTableOracleInput, getCostTableExpected} from './network-cost-table-output-oracles';

for (const stopped of [false, true])
  it(`pins poisoned CSR and isolated/absent convergence (stopped ${stopped})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const withConverged of [false, true]) {
      const expected = getCostTableExpected(stopped);
      const result = await runPoisonedOperation(
        device,
        new GPUNetworkCostTableOperation({
          nodeCount: 3,
          capacity: expected.targets.length,
          laneCount: 2,
          maximumIterations: stopped ? 1 : 8,
          localIterations: 1,
          topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          costLimit: 'cutoff',
          output: {
            offsets: 'outputOffsets',
            targets: 'targets',
            costs: 'costs',
            status: withConverged ? {converged: 'converged'} : undefined
          }
        }),
        {
          offsets: {format: 'uint32', values: costTableOracleInput.topology.offsets},
          neighbors: {format: 'uint32', values: costTableOracleInput.topology.neighbors},
          weights: {format: 'float32', values: costTableOracleInput.topology.weights},
          cutoff: {format: 'float32', values: costTableOracleInput.costLimit}
        }
      );
      expectOracleRows(result['outputOffsets'], expected.offsets);
      expectOracleRows(result['targets'], expected.targets);
      expectOracleRows(result['costs'], expected.costs);
      if (withConverged) expectOracleRows(result['converged'], expected.converged);
    }
  });
