// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CPUSpatialWeightsValidationOperation,
  computeSpatialWeightsValidationOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';

it('delegates GPU input-shape validation to contributor construction at lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        port === 'weights' ? 'float32' : 'uint32',
        port === 'weights' ? 1 : port === 'invalidCount' ? 1 : 2
      ) as GraphDataView<Format>;
    const operation = new GPUSpatialWeightsValidationOperation({
      id: 'gpu-validation',
      rowCount: 1,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {status: {invalidCount: 'invalid'}}
    });
    expect(() => operation.createContributor(getView)).toThrow(
      /gpu-validation weights.weights length/
    );
  } finally {
    device.destroy();
  }
});

it('matches manually assigned violation words for every bit and combinations', () => {
  // Row 0: self, duplicate and invalid weight (8|4|16). Row 1: valid zero weight.
  // Row 2: out-of-range neighbor and negative weight (2|16). Row 3: self (8).
  const result = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(0, 3, 4, 5, 6),
    Uint32Array.of(0, 1, 1, 2, 9, 3),
    Float32Array.of(1, 1, NaN, 0, -1, 1)
  );
  expect(Array.from(result.flags)).toEqual([28, 0, 18, 8]);
  expect(result.invalidCount[0]).toBe(3);
  const offsetViolations = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(1, 0, 9),
    Uint32Array.of(0, 0),
    Float32Array.of(1, 1)
  );
  // Row 0 decreases and starts at 1. Row 1 overflows and lists duplicate zero neighbors.
  expect(Array.from(offsetViolations.flags)).toEqual([1, 5]);
  expect(offsetViolations.invalidCount[0]).toBe(2);
});

it('accepts zero-capacity CSR as an extension: a single empty row has no violation', () => {
  const result = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(0, 0),
    new Uint32Array(),
    new Float32Array()
  );
  expect(Array.from(result.flags)).toEqual([0]);
  expect(Array.from(result.invalidCount)).toEqual([0]);
});

it('accepts a known valid CSR, ignores unused slots, and never reads beyond capacity', () => {
  const result = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(0, 2, 3, 4),
    Uint32Array.of(1, 2, 0, 0, 0xffffffff),
    Float32Array.of(0.5, 0.5, 1, 1, NaN)
  );
  expect(Array.from(result.flags)).toEqual([0, 0, 0]);
  expect(result.invalidCount[0]).toBe(0);
  expect(
    Array.from(
      computeSpatialWeightsValidationOnCPU(
        Uint32Array.of(0, 0, 0),
        new Uint32Array(),
        new Float32Array()
      ).flags
    )
  ).toEqual([0, 0]);
  expect(
    Array.from(
      computeSpatialWeightsValidationOnCPU(
        Uint32Array.of(0, 0xffffffff, 0xffffffff),
        Uint32Array.of(1),
        Float32Array.of(1)
      ).flags
    )
  ).toEqual([0, 1]);
  expect(() =>
    computeSpatialWeightsValidationOnCPU(Uint32Array.of(0), Uint32Array.of(0), Float32Array.of(1))
  ).toThrow('offsets');
  expect(() =>
    computeSpatialWeightsValidationOnCPU(
      Uint32Array.of(0, 1),
      Uint32Array.of(0),
      new Float32Array()
    )
  ).toThrow('weights');
});

it('validates construction and declares identical outputs, including optional flags', () => {
  const props = {
    id: 'validate-test',
    rowCount: 3,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {flags: 'flags', status: {invalidCount: 'invalid'}}
  };
  expect(() => new CPUSpatialWeightsValidationOperation({...props, rowCount: 0})).toThrow(
    /validate-test rowCount/
  );
  expect(() => new GPUSpatialWeightsValidationOperation({...props, rowCount: 0})).toThrow(
    /validate-test outputs.flags.length/
  );
  for (const flags of ['flags', undefined]) {
    const options = {...props, output: {...props.output, flags}};
    expect(new CPUSpatialWeightsValidationOperation(options).contract.outputs).toEqual(
      new GPUSpatialWeightsValidationOperation(options).contract.outputs
    );
  }
});
