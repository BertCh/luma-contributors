// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRadiationFlowModelOnCPU,
  CPURadiationFlowModelOperation
} from '../../../src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation';
import {GPURadiationFlowModelOperation} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model-operation';
import type {
  GPURadiationFlowModel,
  GPURadiationFlowModelNormalization
} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

function makeCPUVector(format: GPUCPUVector['format'], values: Float32Array): GPUCPUVector {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}

it.each([
  null,
  undefined,
  'none',
  'finite-size'
] as const)('S2 resolves normalization consistently for JavaScript value %s', async normalization => {
  // JavaScript callers can supply null although the public TypeScript option remains optional.
  const option = normalization as GPURadiationFlowModelNormalization | undefined;
  const input = {
    positions: Float32Array.of(0, 0, 1, 0),
    populations: Float32Array.of(5, 1),
    outflows: Float32Array.of(12, 9),
    normalization: option
  };
  // Without correction: 12*5*1/(5*6)=2 and 9*1*5/(1*6)=7.5. With one
  // destination per origin, finite-size correction assigns the entire outflow to it.
  const expected = normalization === 'finite-size' ? [0, 12, 9, 0] : [0, 2, 7.5, 0];
  expect(Array.from(computeRadiationFlowModelOnCPU(input).flows)).toEqual(expected);
  const props = {
    zoneCount: 2,
    positions: 'positions',
    populations: 'populations',
    outflows: 'outflows',
    normalization: option,
    output: {flows: 'flows'}
  };
  const gpuOperation = new GPURadiationFlowModelOperation(props);
  try {
    const contributor = constructNetworkOperationContributor(gpuOperation, {
      positions: {format: 'float32x2', length: 2},
      populations: {format: 'float32', length: 2},
      outflows: {format: 'float32', length: 2}
    }) as GPURadiationFlowModel;
    expect(contributor.props.normalization).toBe(normalization ?? 'none');
  } finally {
    gpuOperation.destroy();
  }
  const rows = await new CPURadiationFlowModelOperation(props).execute({
    inputs: {
      positions: makeCPUVector('float32x2', input.positions),
      populations: makeCPUVector('float32', input.populations),
      outflows: makeCPUVector('float32', input.outflows)
    }
  });
  expect(rows['flows']).toEqual(Float32Array.from(expected));
});
