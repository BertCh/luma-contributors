// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUNetworkCentralityOperation} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectOracleRows, runPoisonedOperation} from '../alternative-routes/output-oracle-harness';
import {
  centralityOracleExpected,
  centralityOracleGraphInputs,
  createCentralityOracleProps,
  createCentralityBudgetOracle,
  expectCentralityOracleOutputs
} from './network-centrality-output-oracles';

it('pins every poisoned diamond centrality output together and in isolation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const selected of ['all', ...Object.keys(centralityOracleExpected)]) {
    const props = createCentralityOracleProps();
    for (const port of Object.keys(
      centralityOracleExpected
    ) as (keyof typeof centralityOracleExpected)[])
      if (selected === 'all' || selected === port) props.output[port] = port;
    const result = await runPoisonedOperation(
      device,
      new GPUNetworkCentralityOperation(props),
      centralityOracleGraphInputs
    );
    expectCentralityOracleOutputs(selected, result);
  }
});

for (const [betweenness, rounds] of [
  [false, 1],
  [true, 1],
  [true, 2],
  [true, 3]
] as const)
  it(`R2 pins centrality search/dependency rounds (betweenness ${betweenness}, rounds ${rounds})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {props, inputs, expected} = createCentralityBudgetOracle(betweenness, rounds);
    const result = await runPoisonedOperation(
      device,
      new GPUNetworkCentralityOperation(props),
      inputs
    );
    for (const [port, rows] of Object.entries(expected))
      expectOracleRows(result[port], rows, port === 'closeness' ? 2e-5 : 0);
  });
