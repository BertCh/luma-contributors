// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUNetworkLouvainOperation} from '../../../src/gpu-network/network-optimization/gpu-network-louvain-operation';
import {GPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import {GPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectOracleRows, runPoisonedOperation} from '../alternative-routes/output-oracle-harness';
import {
  createLouvainOutputOracle,
  createRationalOutputOracle,
  createTimeRoutingOutputOracle,
  rationalStatusPorts,
  rationalOracleScenes
} from './network-optimization-output-oracles';

for (const sparse of [false, true])
  for (const bounded of [false, true])
    it(`pins poisoned Louvain approximation alone and absent (sparse ${sparse}, bounded ${bounded})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const withApproximation of [false, true]) {
        const {props, inputs, expected} = createLouvainOutputOracle(
          sparse,
          bounded,
          withApproximation
        );
        const result = await runPoisonedOperation(
          device,
          new GPUNetworkLouvainOperation(props),
          inputs
        );
        for (const [port, values] of Object.entries(expected))
          expectOracleRows(result[port], values);
        if (withApproximation) expectOracleRows(result['approximation'], [1]);
      }
    });

for (const scene of rationalOracleScenes)
  it(`pins rational-access poisoned status individually and absent (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const selected of [...rationalStatusPorts, 'absent'] as const) {
      const {props, inputs, expected, status} = createRationalOutputOracle(scene, selected);
      const result = await runPoisonedOperation(
        device,
        new GPUNetworkRationalAccessOperation(props),
        inputs
      );
      // Only the 5/6 weighted score is non-dyadic: allow 2e-5 absolute error in float32.
      for (const [port, values] of Object.entries(expected))
        expectOracleRows(result[port], values, port === 'scores' ? 2e-5 : 0);
      if (selected !== 'absent') expectOracleRows(result[selected], status[selected]);
    }
  });

for (const piecewise of [false, true])
  for (const scene of ['stable', 'bounded', 'invalid'] as const)
    it(`pins poisoned time-routing parents and packed summary (piecewise ${piecewise}, ${scene})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const withInvalidCount of [false, true]) {
        const {props, inputs, expected, invalidCount} = createTimeRoutingOutputOracle(
          scene,
          piecewise,
          withInvalidCount
        );
        const result = await runPoisonedOperation(
          device,
          new GPUNetworkTimeRoutingOperation(props),
          inputs
        );
        // All interpolation fractions and durations are dyadic, so labels and packed bits are exact.
        for (const [port, values] of Object.entries(expected))
          expectOracleRows(result[port], values);
        if (withInvalidCount) expectOracleRows(result['invalidCount'], invalidCount);
      }
    });
