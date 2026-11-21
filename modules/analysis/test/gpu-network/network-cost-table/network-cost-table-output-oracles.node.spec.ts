// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {it} from 'vitest';
import {computeNetworkCostTableOnCPU} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {expectOracleRows} from '../alternative-routes/output-oracle-harness';
import {costTableOracleInput, getCostTableExpected} from './network-cost-table-output-oracles';

for (const stopped of [false, true])
  it(`pins enumerated chain CSR and convergence (stopped ${stopped})`, () => {
    const expected = getCostTableExpected(stopped);
    const result = computeNetworkCostTableOnCPU(costTableOracleInput, {
      capacity: expected.targets.length,
      maximumIterations: stopped ? 1 : 8,
      localIterations: 1
    });
    for (const [port, values] of Object.entries(expected))
      expectOracleRows(result[port as keyof typeof result], values);
  });
