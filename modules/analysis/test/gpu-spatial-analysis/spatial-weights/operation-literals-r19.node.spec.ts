// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUCPUOperation, GPUCPUResult} from '@luma.gl/graph/gpu-core';
import {CPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {CPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {
  ALGEBRA_EXPECTATIONS,
  ALGEBRA_LEFT,
  ALGEBRA_PORTS,
  ALGEBRA_RIGHT,
  CONTIGUITY_EXPECTATIONS,
  CONTIGUITY_INPUT,
  CONTIGUITY_PORTS,
  cpuVector,
  VALIDATION_EXPECTED,
  VALIDATION_INPUT,
  WEIGHTS_OUTPUT
} from './operation-literals-r19.test-utils';

/** Exercise the asynchronous return permitted by the installed operation contract. */
function deferExecution<Operation extends GPUCPUOperation>(operation: Operation): Operation {
  const execute = operation.execute;
  Object.defineProperty(operation, 'execute', {
    value: (context: Parameters<typeof execute>[0]) => Promise.resolve(execute(context))
  });
  return operation;
}

function uint32Result(result: GPUCPUResult | undefined): Uint32Array {
  if (!(result instanceof Uint32Array)) throw new Error('expected a packed Uint32Array result');
  return result;
}

function float32Result(result: GPUCPUResult | undefined): Float32Array {
  if (!(result instanceof Float32Array)) throw new Error('expected a packed Float32Array result');
  return result;
}

for (const fixture of CONTIGUITY_EXPECTATIONS)
  it(`Tests §5 CPU contiguity ${fixture.criterion} matches independent edge/corner literals`, async () => {
    const operation = deferExecution(
      new CPUContiguityWeightsOperation({
        criterion: fixture.criterion,
        featureCount: 3,
        capacity: 4,
        pairCapacity: 64,
        polygons: CONTIGUITY_PORTS,
        output: WEIGHTS_OUTPUT
      })
    );
    const result = await operation.execute({
      inputs: {
        positions: {
          ...cpuVector(CONTIGUITY_INPUT.positions),
          format: 'float32x2',
          length: 12,
          chunks: [{values: CONTIGUITY_INPUT.positions, length: 12}]
        },
        ringOffsets: cpuVector(CONTIGUITY_INPUT.ringOffsets),
        polygonOffsets: cpuVector(CONTIGUITY_INPUT.polygonOffsets),
        featureOffsets: cpuVector(CONTIGUITY_INPUT.featureOffsets)
      }
    });
    expect(Array.from(uint32Result(result['offsets']))).toEqual(fixture.offsets);
    expect(
      Array.from(uint32Result(result['neighbors'])).slice(0, fixture.neighbors.length)
    ).toEqual(fixture.neighbors);
    expect(Array.from(uint32Result(result['requiredCount']))).toEqual([fixture.neighbors.length]);
    expect(Array.from(uint32Result(result['overflow']))).toEqual([0]);
    expect(Array.from(uint32Result(result['candidateOverflow']))).toEqual([0]);
  });

for (const fixture of ALGEBRA_EXPECTATIONS)
  it(`Tests §5 CPU ${fixture.operation} matches hand-listed directed edges`, async () => {
    const operation = deferExecution(
      new CPUSpatialWeightsAlgebraOperation({
        rowCount: 3,
        capacity: 6,
        operation: fixture.operation,
        ...ALGEBRA_PORTS,
        output: WEIGHTS_OUTPUT
      })
    );
    const result = await operation.execute({
      inputs: {
        leftOffsets: cpuVector(ALGEBRA_LEFT.offsets),
        leftNeighbors: cpuVector(ALGEBRA_LEFT.neighbors),
        leftWeights: cpuVector(ALGEBRA_LEFT.weights),
        rightOffsets: cpuVector(ALGEBRA_RIGHT.offsets),
        rightNeighbors: cpuVector(ALGEBRA_RIGHT.neighbors),
        rightWeights: cpuVector(ALGEBRA_RIGHT.weights)
      }
    });
    expect(Array.from(uint32Result(result['offsets']))).toEqual(fixture.offsets);
    expect(
      Array.from(uint32Result(result['neighbors'])).slice(0, fixture.neighbors.length)
    ).toEqual(fixture.neighbors);
    expect(Array.from(uint32Result(result['requiredCount']))).toEqual([fixture.neighbors.length]);
    expect(Array.from(float32Result(result['weights'])).slice(0, fixture.neighbors.length)).toEqual(
      Array(fixture.neighbors.length).fill(1)
    );
  });

it('Tests §5 CPU validation counts one row with two independent violations', async () => {
  const result = await deferExecution(
    new CPUSpatialWeightsValidationOperation({
      rowCount: 3,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {flags: 'flags', status: {invalidCount: 'invalidCount'}}
    })
  ).execute({
    inputs: {
      offsets: cpuVector(VALIDATION_INPUT.offsets),
      neighbors: cpuVector(VALIDATION_INPUT.neighbors),
      weights: cpuVector(VALIDATION_INPUT.weights)
    }
  });
  expect(Array.from(uint32Result(result['flags']))).toEqual(VALIDATION_EXPECTED.flags);
  expect(Array.from(uint32Result(result['invalidCount']))).toEqual(
    VALIDATION_EXPECTED.invalidCount
  );
});
