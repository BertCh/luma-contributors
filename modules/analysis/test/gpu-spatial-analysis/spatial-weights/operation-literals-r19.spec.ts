// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {CPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {CPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {GPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  ALGEBRA_EXPECTATIONS,
  ALGEBRA_LEFT,
  ALGEBRA_PORTS,
  ALGEBRA_RIGHT,
  CONTIGUITY_EXPECTATIONS,
  CONTIGUITY_INPUT,
  CONTIGUITY_PORTS,
  VALIDATION_EXPECTED,
  VALIDATION_INPUT,
  WEIGHTS_OUTPUT
} from './operation-literals-r19.test-utils';

for (const fixture of CONTIGUITY_EXPECTATIONS)
  it(`Tests §5 both contiguity operations match ${fixture.criterion} literals`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [CPUContiguityWeightsOperation, GPUContiguityWeightsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: CONTIGUITY_INPUT.positions},
          ringOffsets: {format: 'uint32', values: CONTIGUITY_INPUT.ringOffsets},
          polygonOffsets: {format: 'uint32', values: CONTIGUITY_INPUT.polygonOffsets},
          featureOffsets: {format: 'uint32', values: CONTIGUITY_INPUT.featureOffsets}
        },
        operations: [
          new Operation({
            criterion: fixture.criterion,
            featureCount: 3,
            capacity: 4,
            pairCapacity: 64,
            polygons: CONTIGUITY_PORTS,
            output: WEIGHTS_OUTPUT
          })
        ],
        outputs: ['offsets', 'neighbors', 'requiredCount', 'overflow', 'candidateOverflow']
      });
      expect(Array.from(result['offsets'])).toEqual(fixture.offsets);
      expect(Array.from(result['neighbors']).slice(0, fixture.neighbors.length)).toEqual(
        fixture.neighbors
      );
      expect(Array.from(result['requiredCount'])).toEqual([fixture.neighbors.length]);
      expect(Array.from(result['overflow'])).toEqual([0]);
      expect(Array.from(result['candidateOverflow'])).toEqual([0]);
    }
  });

for (const fixture of ALGEBRA_EXPECTATIONS)
  it(`Tests §5 both algebra operations match ${fixture.operation} literals`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      CPUSpatialWeightsAlgebraOperation,
      GPUSpatialWeightsAlgebraOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          leftOffsets: {format: 'uint32', values: ALGEBRA_LEFT.offsets},
          leftNeighbors: {format: 'uint32', values: ALGEBRA_LEFT.neighbors},
          leftWeights: {format: 'float32', values: ALGEBRA_LEFT.weights},
          rightOffsets: {format: 'uint32', values: ALGEBRA_RIGHT.offsets},
          rightNeighbors: {format: 'uint32', values: ALGEBRA_RIGHT.neighbors},
          rightWeights: {format: 'float32', values: ALGEBRA_RIGHT.weights}
        },
        operations: [
          new Operation({
            rowCount: 3,
            capacity: 6,
            operation: fixture.operation,
            ...ALGEBRA_PORTS,
            output: WEIGHTS_OUTPUT
          })
        ],
        outputs: ['offsets', 'neighbors', 'weights', 'requiredCount']
      });
      expect(Array.from(result['offsets'])).toEqual(fixture.offsets);
      expect(Array.from(result['neighbors']).slice(0, fixture.neighbors.length)).toEqual(
        fixture.neighbors
      );
      expect(Array.from(result['weights']).slice(0, fixture.neighbors.length)).toEqual(
        Array(fixture.neighbors.length).fill(1)
      );
      expect(Array.from(result['requiredCount'])).toEqual([fixture.neighbors.length]);
    }
  });

it('Tests §5 both validation operations count one invalid row rather than two violations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    CPUSpatialWeightsValidationOperation,
    GPUSpatialWeightsValidationOperation
  ]) {
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: VALIDATION_INPUT.offsets},
        neighbors: {format: 'uint32', values: VALIDATION_INPUT.neighbors},
        weights: {format: 'float32', values: VALIDATION_INPUT.weights}
      },
      operations: [
        new Operation({
          rowCount: 3,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          output: {flags: 'flags', status: {invalidCount: 'invalidCount'}}
        })
      ],
      outputs: ['flags', 'invalidCount']
    });
    expect(Array.from(result['flags'])).toEqual(VALIDATION_EXPECTED.flags);
    expect(Array.from(result['invalidCount'])).toEqual(VALIDATION_EXPECTED.invalidCount);
  }
});
