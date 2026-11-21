// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUNetworkCostMatrixOperation} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectOracleRows, runPoisonedOperation} from '../alternative-routes/output-oracle-harness';
import {
  createCostMatrixOutputOracle,
  createCostMatrixSpillOracle,
  createCostMatrixDefaultOracle,
  createCostMatrixRewriteOracle
} from './network-cost-matrix-output-oracles';

it('R4 pins exactly sixteen default local hops', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {props, inputs, expected} = createCostMatrixDefaultOracle();
  const result = await runPoisonedOperation(
    device,
    new GPUNetworkCostMatrixOperation(props),
    inputs
  );
  for (const [port, rows] of Object.entries(expected)) expectOracleRows(result[port], rows);
});

it('R9 rewrites the seed between submissions on the same compiled matrix graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {props, inputs, first, second} = createCostMatrixRewriteOracle();
  await runPoisonedOperation(device, new GPUNetworkCostMatrixOperation(props), inputs, {
    rewriteInputs: {starts: Uint32Array.of(2)},
    expected: [first, second]
  });
});

it('R5 keeps construction-time matrix budgets after caller mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {props, inputs} = createCostMatrixOutputOracle(false);
  props.maximumIterations = 1;
  props.output.status = {converged: 'converged', iterationLimitReached: 'limited'};
  const operation = new GPUNetworkCostMatrixOperation(props);
  props.maximumIterations = 8;
  props.output.status.converged = 'changed';
  const result = await runPoisonedOperation(device, operation, inputs);
  expectOracleRows(result['costs'], [
    1,
    3,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    0.5,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    Infinity
  ]);
  expectOracleRows(result['converged'], [0]);
  expectOracleRows(result['limited'], [1]);
});

for (const gather of [false, true])
  it(`pins poisoned chain matrix with isolated and absent status ports (gather ${gather})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const status of ['converged', 'iterationLimitReached', 'absent'] as const) {
      const {props, inputs, expected} = createCostMatrixOutputOracle(gather);
      if (status !== 'absent') props.output.status = {[status]: status};
      const result = await runPoisonedOperation(
        device,
        new GPUNetworkCostMatrixOperation(props),
        inputs
      );
      // All path additions are exact integers/binary fractions, including the gathered duplicates.
      expectOracleRows(result['costs'], expected);
      if (status !== 'absent') expectOracleRows(result[status], [status === 'converged' ? 1 : 0]);
    }
  });

for (const neighborCount of [256, 257] as const)
  it(`R1 pins local queue spill across matrix rounds (${neighborCount} neighbors)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {props, inputs, expected} = createCostMatrixSpillOracle(neighborCount);
    const result = await runPoisonedOperation(
      device,
      new GPUNetworkCostMatrixOperation(props),
      inputs
    );
    for (const [port, rows] of Object.entries(expected)) expectOracleRows(result[port], rows);
  });

it('pins nonzero matrix round exhaustion with either status port alone or absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const status of ['converged', 'iterationLimitReached', 'absent'] as const) {
    const {props, inputs} = createCostMatrixOutputOracle(false);
    props.maximumIterations = 1;
    if (status !== 'absent') props.output.status = {[status]: status};
    const result = await runPoisonedOperation(
      device,
      new GPUNetworkCostMatrixOperation(props),
      inputs
    );
    expectOracleRows(result['costs'], [
      1,
      3,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      0.5,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity
    ]);
    if (status !== 'absent')
      expectOracleRows(result[status], [status === 'iterationLimitReached' ? 1 : 0]);
  }
});
