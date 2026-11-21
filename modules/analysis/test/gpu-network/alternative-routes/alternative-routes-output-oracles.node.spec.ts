// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {it} from 'vitest';
import {
  computeAlternativeRoutesOnCPU,
  CPUAlternativeRoutesOperation
} from '../../../src/gpu-network/alternative-routes/cpu-alternative-routes-operation';
import {GPUAlternativeRoutesOperation} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import {
  alternativeRoutesExpected,
  alternativeRoutesInput,
  createAlternativeBudgetOracle
} from './alternative-routes-output-oracles';
import {expectOracleRows, getOracleCPUVectors} from './output-oracle-harness';

it('pins both diamond routes, unreachable query and all padded tails by hand', () => {
  const result = computeAlternativeRoutesOnCPU(alternativeRoutesInput, {routeCount: 3});
  for (const port of Object.keys(
    alternativeRoutesExpected
  ) as (keyof typeof alternativeRoutesExpected)[])
    expectOracleRows(result[port], alternativeRoutesExpected[port]);
});

for (const scene of ['expansion', 'frontier', 'retained'] as const)
  it(`R3 binds alternative-route budgets (${scene})`, async () => {
    const {props, inputs, expected} = createAlternativeBudgetOracle(scene);
    const metadata = new GPUAlternativeRoutesOperation(props);
    const result = await new CPUAlternativeRoutesOperation(props).execute({
      inputs: getOracleCPUVectors(inputs, metadata.inputs)
    });
    for (const [port, rows] of Object.entries(expected))
      expectOracleRows(result[port] as Float32Array | Uint32Array, rows);
  });
