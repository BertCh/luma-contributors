// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRadiationFlowModelOnCPU,
  CPURadiationFlowModelOperation
} from '../../../src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation';
import {
  GPURadiationFlowModelOperation,
  type RadiationFlowModelOperationProps
} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: RadiationFlowModelOperationProps = {
  id: 'radiation-test',
  zoneCount: 3,
  positions: 'positions',
  populations: 'populations',
  outflows: 'outflows',
  output: {flows: 'flows'}
};

it('preserves strict distance ties in a hand-computed equal-population example', () => {
  const result = computeRadiationFlowModelOnCPU({
    positions: new Float32Array([0, 0, 1, 0, -1, 0]),
    populations: new Float32Array([1, 1, 1]),
    outflows: new Float32Array([1, 1, 1])
  });
  // Origin 0 sees two equally near destinations; neither intervenes for the other.
  expect(result.flows).toEqual(
    new Float32Array([0, 1 / 2, 1 / 2, 1 / 2, 0, 1 / 6, 1 / 2, 1 / 6, 0])
  );
});

it('satisfies the telescoping row-sum identity and its finite-size correction', () => {
  const input = {
    positions: new Float32Array([0, 0, 1, 0, 4, 0, 10, 0]),
    populations: new Float32Array([10, 20, 30, 40]),
    outflows: new Float32Array([3, 7, 11, 19])
  };
  for (const normalization of ['none', 'finite-size'] as const) {
    const {flows} = computeRadiationFlowModelOnCPU({...input, normalization});
    for (let origin = 0; origin < 4; origin++) {
      const rowSum = flows
        .subarray(origin * 4, origin * 4 + 4)
        .reduce((sum, flow) => sum + flow, 0);
      expect(rowSum).toBeCloseTo(
        input.outflows[origin] *
          (normalization === 'none' ? 1 - input.populations[origin] / 100 : 1),
        5
      );
    }
  }
});

it('uses great-circle distance ordering and handles zero populations', () => {
  // Across the date line, longitude -179 is nearer to 179 than longitude 170.
  const {flows} = computeRadiationFlowModelOnCPU({
    positions: new Float32Array([179, 0, -179, 0, 170, 0]),
    populations: new Float32Array([1, 1, 1]),
    outflows: new Float32Array([1, 1, 1]),
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}
  });
  expect(flows[1]).toBe(0.5);
  expect(flows[2]).toBeCloseTo(1 / 6, 7);
  expect(
    computeRadiationFlowModelOnCPU({
      positions: new Float32Array(6),
      populations: new Float32Array(3),
      outflows: new Float32Array([1, 1, 1]),
      normalization: 'finite-size'
    }).flows
  ).toEqual(new Float32Array(9));
});

it('keeps tiny representable longitude separations distinct in great-circle ordering', () => {
  const {flows} = computeRadiationFlowModelOnCPU({
    positions: new Float32Array([0, 0, 2 ** -48, 0, 2 ** -47, 0]),
    populations: new Float32Array([1, 1, 1]),
    outflows: new Float32Array([1, 1, 1]),
    normalization: 'none',
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}
  });
  // End origins see one intervening zone before the far destination: 1 / ((1+1)(1+1+1)).
  // The middle origin has two equally near destinations, so neither intervenes for the other.
  expect(flows).toEqual(new Float32Array([0, 1 / 2, 1 / 6, 1 / 2, 0, 1 / 2, 1 / 6, 1 / 2, 0]));
});

it('rejects inconsistent arrays and validates both operation constructions', () => {
  expect(() =>
    computeRadiationFlowModelOnCPU({
      positions: new Float32Array(2),
      populations: new Float32Array(3),
      outflows: new Float32Array(3)
    })
  ).toThrow(/positions/);
  expect(() =>
    computeRadiationFlowModelOnCPU({
      positions: new Float32Array(6),
      populations: new Float32Array(3),
      outflows: new Float32Array(2)
    })
  ).toThrow(/outflows/);
  expect(() => new CPURadiationFlowModelOperation({...PROPS, zoneCount: 1})).toThrow(
    /radiation-test.*zoneCount/
  );
  const invalid = {...PROPS, normalization: 'invalid' as 'none'};
  expect(() => new CPURadiationFlowModelOperation(invalid)).toThrow(
    /radiation-test.*normalization/
  );
  const shapes = {
    positions: {format: 'float32x2' as const, length: 3},
    populations: {format: 'float32' as const, length: 3},
    outflows: {format: 'float32' as const, length: 3}
  };
  expect(() =>
    constructNetworkOperationContributor(new GPURadiationFlowModelOperation(invalid), shapes)
  ).toThrow(/radiation-test.*normalization/);
  expect(() =>
    constructNetworkOperationContributor(new GPURadiationFlowModelOperation(PROPS), shapes)
  ).not.toThrow();
  for (const Operation of [GPURadiationFlowModelOperation, CPURadiationFlowModelOperation]) {
    expect(() => new Operation({...PROPS, output: {flows: 'positions'}})).toThrow(/radiation-test/);
  }
  // The GPU work cap is deliberately absent on the CPU.
  expect(() => new CPURadiationFlowModelOperation({...PROPS, zoneCount: 1025})).not.toThrow();
});
