// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUAlternativeRoutesOperation} from '../../../src/gpu-network/alternative-routes/gpu-alternative-routes-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  alternativeRoutesExpected,
  alternativeRoutesInput,
  createAlternativeBudgetOracle
} from './alternative-routes-output-oracles';
import {expectOracleRows, runPoisonedOperation} from './output-oracle-harness';

it('pins both diamond routes, unreachable query and poisoned padded tails by hand', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runPoisonedOperation(
    device,
    new GPUAlternativeRoutesOperation({
      nodeCount: 4,
      queryCount: 2,
      routeCount: 3,
      maximumFrontierSize: 8,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      sources: 'sources',
      targets: 'targets',
      output: {
        vertices: 'vertices',
        edges: 'edges',
        lengths: 'lengths',
        costs: 'costs',
        counts: 'counts',
        complete: 'complete'
      }
    }),
    {
      offsets: {format: 'uint32', values: alternativeRoutesInput.topology.offsets},
      neighbors: {format: 'uint32', values: alternativeRoutesInput.topology.neighbors},
      weights: {format: 'float32', values: alternativeRoutesInput.topology.weights},
      sources: {format: 'uint32', values: alternativeRoutesInput.sources},
      targets: {format: 'uint32', values: alternativeRoutesInput.targets}
    }
  );
  for (const [port, expected] of Object.entries(alternativeRoutesExpected))
    expectOracleRows(result[port], expected);
});

for (const scene of ['expansion', 'frontier', 'retained'] as const)
  it(`R3 pins alternative-route budgets (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {props, inputs, expected} = createAlternativeBudgetOracle(scene);
    const result = await runPoisonedOperation(
      device,
      new GPUAlternativeRoutesOperation(props),
      inputs
    );
    for (const [port, rows] of Object.entries(expected)) expectOracleRows(result[port], rows);
  });
