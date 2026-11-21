// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  computeNetworkCentralityOnCPU,
  CPUNetworkCentralityOperation
} from '../../../src/gpu-network/network-centrality/cpu-network-centrality-operation';
import {GPUNetworkCentralityOperation} from '../../../src/gpu-network/network-centrality/gpu-network-centrality-operation';
import {expectOracleRows, getOracleCPUVectors} from '../alternative-routes/output-oracle-harness';
import {
  centralityOracleExpected,
  centralityOracleInput,
  createCentralityBudgetOracle,
  expectCentralityOracleOutputs
} from './network-centrality-output-oracles';

it('pins scaled diamond dependencies and per-pivot outward closeness by hand', () => {
  const result = computeNetworkCentralityOnCPU(centralityOracleInput, {
    betweennessScale: 2,
    sampling: 'sampled'
  });
  for (const [port, expected] of Object.entries(centralityOracleExpected))
    expectOracleRows(result[port as keyof typeof result], expected);
});

it('R6 rejects a renamed or missing isolated closeness graph value', () => {
  expect(() =>
    expectCentralityOracleOutputs('closeness', {renamedCloseness: [0.75, 0, 0]})
  ).toThrow();
  expect(() => expectCentralityOracleOutputs('closeness', {})).toThrow();
  expect(() => expectCentralityOracleOutputs('closeness', {closeness: [0.75, 0, 0]})).not.toThrow();
});

for (const [betweenness, rounds] of [
  [false, 1],
  [true, 1],
  [true, 2],
  [true, 3]
] as const)
  it(`R2 binds centrality search/dependency rounds (betweenness ${betweenness}, rounds ${rounds})`, async () => {
    const {props, inputs, expected} = createCentralityBudgetOracle(betweenness, rounds);
    const metadata = new GPUNetworkCentralityOperation(props);
    const result = await new CPUNetworkCentralityOperation(props).execute({
      inputs: getOracleCPUVectors(inputs, metadata.inputs)
    });
    for (const [port, rows] of Object.entries(expected))
      expectOracleRows(
        result[port] as Uint32Array | Float32Array,
        rows,
        port === 'closeness' ? 2e-5 : 0
      );
  });
