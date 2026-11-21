// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {CPUNetworkLouvainOperation} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
import {CPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/cpu-network-rational-access-operation';
import {CPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {expectOracleRows, getOracleCPUVectors} from '../alternative-routes/output-oracle-harness';
import {
  createLouvainOutputOracle,
  createRationalOutputOracle,
  createTimeRoutingOutputOracle,
  rationalStatusPorts,
  rationalOracleScenes
} from './network-optimization-output-oracles';

for (const sparse of [false, true])
  for (const bounded of [false, true])
    it(`pins Louvain heuristic approximation on converged/bounded pair (sparse ${sparse}, bounded ${bounded})`, async () => {
      for (const withApproximation of [false, true]) {
        const {props, inputs, expected} = createLouvainOutputOracle(
          sparse,
          bounded,
          withApproximation
        );
        const result = await new CPUNetworkLouvainOperation(props).execute({
          inputs: getOracleCPUVectors(inputs)
        });
        for (const [port, values] of Object.entries(expected))
          expectOracleRows(result[port] as Uint32Array | Float32Array, values);
        if (withApproximation) expectOracleRows(result['approximation'] as Uint32Array, [1]);
      }
    });

it('R8 distinguishes rational iteration count from validity', () => {
  expect(
    rationalOracleScenes.map(
      scene => createRationalOutputOracle(scene, 'iterationCount').status.iterationCount[0]
    )
  ).toContain(2);
});

for (const scene of rationalOracleScenes)
  it(`pins rational-access isolated status and allocations by hand (${scene})`, async () => {
    for (const selected of [...rationalStatusPorts, 'absent'] as const) {
      const {props, inputs, expected, status} = createRationalOutputOracle(scene, selected);
      const result = await new CPUNetworkRationalAccessOperation(props).execute({
        inputs: getOracleCPUVectors(inputs)
      });
      for (const [port, values] of Object.entries(expected))
        expectOracleRows(
          result[port] as Uint32Array | Float32Array,
          values,
          port === 'scores' ? 2e-5 : 0
        );
      if (selected !== 'absent')
        expectOracleRows(result[selected] as Uint32Array, status[selected]);
    }
  });

for (const piecewise of [false, true])
  for (const scene of ['stable', 'bounded', 'invalid'] as const)
    it(`pins time-routing parent rows, target bits and invalid flag (piecewise ${piecewise}, ${scene})`, async () => {
      for (const withInvalidCount of [false, true]) {
        const {props, inputs, expected, invalidCount} = createTimeRoutingOutputOracle(
          scene,
          piecewise,
          withInvalidCount
        );
        const result = await new CPUNetworkTimeRoutingOperation(props).execute({
          inputs: getOracleCPUVectors(inputs)
        });
        for (const [port, values] of Object.entries(expected))
          expectOracleRows(result[port] as Uint32Array | Float32Array, values);
        if (withInvalidCount) expectOracleRows(result['invalidCount'] as Uint32Array, invalidCount);
      }
    });
