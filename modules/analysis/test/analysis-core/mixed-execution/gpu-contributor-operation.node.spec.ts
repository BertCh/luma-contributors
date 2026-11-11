// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationInt32Rows,
  getCPUOperationUint32Rows
} from '../../../src/analysis-core/operations/cpu-operation-rows';
import {
  GPUContributorOperation,
  GPU_CONTRIBUTOR_OPERATION_LOWERING,
  GPU_CONTRIBUTOR_OPERATION_TYPE
} from '../../../src/analysis-core/operations/gpu-contributor-operation';

const NO_CONTRIBUTOR = () => ({getCommandNodes: () => []});

function makeRows(
  format: GPUCPUVector['format'],
  chunks: readonly (Float32Array | Uint32Array | Int32Array)[]
): GPUCPUVector {
  // `length` and each chunk's `length` count rows; the arrays hold `components` values per row.
  const {components} = getGPUVectorFormatInfo(format);
  return {
    format,
    length: chunks.reduce((total, chunk) => total + chunk.length / components, 0),
    chunks: chunks.map(values => ({values, length: values.length / components})),
    pack() {
      const packed = new (chunks[0].constructor as typeof Float32Array)(this.length * components);
      let offset = 0;
      for (const chunk of chunks) {
        packed.set(chunk, offset);
        offset += chunk.length;
      }
      return packed;
    }
  };
}

it('a contributor operation declares its ports as a contract over graph values', () => {
  const operation = new GPUContributorOperation({
    id: 'fit',
    // Two ports may read the same graph value; the contract lists it once.
    inputs: {distances: 'lags', weights: 'lags', counts: 'pairs'},
    outputs: {
      parameters: {value: 'fitted', format: 'float32x4', length: 3},
      status: {value: 'flags', format: 'uint32', length: 3}
    },
    createContributor: NO_CONTRIBUTOR
  });
  expect(operation.type).toBe(GPU_CONTRIBUTOR_OPERATION_TYPE);
  expect(GPU_CONTRIBUTOR_OPERATION_LOWERING.type).toBe(GPU_CONTRIBUTOR_OPERATION_TYPE);
  const {contract} = operation;
  expect(contract.values).toEqual([
    {value: 'lags', access: 'read'},
    {value: 'pairs', access: 'read'},
    {value: 'fitted', access: 'write', shape: {length: 3}},
    {value: 'flags', access: 'write', shape: {length: 3}}
  ]);
  expect(contract.outputs).toEqual({
    fitted: {type: 'vector', format: 'float32x4', length: 3},
    flags: {type: 'vector', format: 'uint32', length: 3}
  });
});

it('a contributor operation rejects ports that cannot be planned, naming the port', () => {
  const output = {value: 'result', format: 'float32', length: 4} as const;
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {values: output},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth port "values" must not be both an input and an output');
  // In place: the published mixed executor has no read-write values.
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: {...output, value: 'values'}},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.result must name a graph value ("values") that no other port uses');
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: output, copy: output},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.copy must name a graph value ("result") that no other port uses');
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: {...output, length: 0}},
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.result.length must be an integer in [1, 4294967295]');
  // A port the operation names as one that may be empty accepts zero rows, and only that port.
  const empty = new GPUContributorOperation({
    id: 'smooth',
    inputs: {values: 'values'},
    outputs: {result: {...output, length: 0}, count: {value: 'count', format: 'uint32', length: 1}},
    emptyOutputPorts: ['result'],
    createContributor: NO_CONTRIBUTOR
  });
  expect(empty.contract.outputs).toEqual({
    result: {type: 'vector', format: 'float32', length: 0},
    count: {type: 'vector', format: 'uint32', length: 1}
  });
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: {...output, length: -1}},
        emptyOutputPorts: ['result'],
        createContributor: NO_CONTRIBUTOR
      })
  ).toThrow('smooth outputs.result.length must be an integer in [0, 4294967295]');
  // A listed name that is not an output port is ignored: optional ports are listed unconditionally.
  expect(
    () =>
      new GPUContributorOperation({
        id: 'smooth',
        inputs: {values: 'values'},
        outputs: {result: output},
        emptyOutputPorts: ['optional'],
        createContributor: NO_CONTRIBUTOR
      })
  ).not.toThrow();
});

it('an operation destroys the resource-owning contributor of the previous run', () => {
  const destroyed: number[] = [];
  let built = 0;
  const operation = new GPUContributorOperation({
    id: 'project',
    inputs: {positions: 'positions'},
    outputs: {result: {value: 'projected', format: 'float32x2', length: 4}},
    createContributor: () => {
      const serial = built++;
      return {getCommandNodes: () => [], destroy: () => destroyed.push(serial)};
    }
  });
  const getView = (() => {
    throw new Error('not resolved in this test');
  }) as Parameters<typeof operation.createContributor>[0];
  operation.createContributor(getView);
  expect(destroyed).toEqual([]);
  // The second run replaces the first run's contributor.
  operation.createContributor(getView);
  expect(destroyed).toEqual([0]);
  operation.destroy();
  operation.destroy();
  expect(destroyed).toEqual([0, 1]);
  // A contributor without resources is not tracked.
  const plain = new GPUContributorOperation({
    id: 'plain',
    inputs: {},
    outputs: {result: {value: 'result', format: 'uint32', length: 1}},
    createContributor: NO_CONTRIBUTOR
  });
  plain.createContributor(getView);
  expect(() => plain.destroy()).not.toThrow();
});

it('the lowering refuses an operation of another class that carries its type', () => {
  expect(() =>
    GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(
      {id: 'impostor', type: GPU_CONTRIBUTOR_OPERATION_TYPE},
      {} as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1]
    )
  ).toThrow('impostor is not a GPUContributorOperation');
});

it('port names are own keys: a name on Object.prototype is an ordinary port', () => {
  // `constructor` is an output port only, not also an input inherited from Object.prototype.
  const operation = new GPUContributorOperation({
    id: 'prototype',
    inputs: {values: 'values'},
    // TypeScript widens the format of a `constructor` key, hence `as const`.
    outputs: {constructor: {value: 'result', format: 'float32' as const, length: 1}},
    createContributor: getView => {
      getView('constructor');
      getView('valueOf');
      return NO_CONTRIBUTOR();
    }
  });
  const resolved: unknown[] = [];
  const context = {
    resolveVector: (value: unknown) => {
      resolved.push(value);
      return {data: [{}]};
    },
    lower: () => {}
  } as unknown as Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1];
  // `valueOf` is no port of the operation, so it is refused by name, not passed on as a function.
  expect(() => GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context)).toThrow(
    'prototype has no port "valueOf"'
  );
  expect(resolved).toEqual(['result']);
});

it('CPU operation row helpers return packed rows and name the port when the input is wrong', () => {
  const single = new Float32Array([1, 2, 3, 4]);
  const inputs = {
    positions: makeRows('float32x2', [single]),
    chunked: makeRows('uint32', [new Uint32Array([1, 2]), new Uint32Array([3])]),
    offsets: makeRows('sint32', [new Int32Array([-1, 5])]),
    chunkedPositions: makeRows('float32x2', [
      new Float32Array([1, 2]),
      new Float32Array([3, 4, 5, 6])
    ]),
    scale: 2
  };
  // One chunk is returned as is, without a second copy.
  expect(getCPUOperationFloat32Rows('twin', inputs, 'positions', 'float32x2')).toBe(single);
  expect(Array.from(getCPUOperationUint32Rows('twin', inputs, 'chunked'))).toEqual([1, 2, 3]);
  // Several chunks of a multi-component format are packed row after row.
  expect(inputs.chunkedPositions.length).toBe(3);
  expect(
    Array.from(getCPUOperationFloat32Rows('twin', inputs, 'chunkedPositions', 'float32x2'))
  ).toEqual([1, 2, 3, 4, 5, 6]);
  expect(Array.from(getCPUOperationInt32Rows('twin', inputs, 'offsets'))).toEqual([-1, 5]);
  expect(() => getCPUOperationFloat32Rows('twin', inputs, 'positions')).toThrow(
    'twin input "positions" must be float32, not float32x2'
  );
  expect(() => getCPUOperationFloat32Rows('twin', inputs, 'scale')).toThrow(
    'twin input "scale" must be a float32 vector'
  );
  expect(() => getCPUOperationUint32Rows('twin', inputs, 'missing')).toThrow(
    'twin input "missing" must be a uint32 vector'
  );
});
