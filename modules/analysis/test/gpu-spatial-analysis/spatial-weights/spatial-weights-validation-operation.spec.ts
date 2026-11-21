// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import {GPUChunkedSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-validation';
import {createRasterTimeSeriesHarness} from '../../gpu-raster/raster-time-series/raster-time-series-test-utils';

it('UST-11 checks declared rectangular and diagonal policies in both operation forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of [
    {neighbors: [0, 1], options: {}, expected: [8, 8]},
    {neighbors: [0, 1], options: {diagonal: 'allow' as const}, expected: [0, 0]},
    {neighbors: [0, 1], options: {diagonal: 'require-unit' as const}, expected: [0, 0]},
    {neighbors: [2, 0], options: {}, expected: [2, 0]},
    {neighbors: [2, 0], options: {columnCount: 3}, expected: [0, 0]},
    {
      neighbors: [2, 0],
      options: {columnCount: 3, diagonal: 'require-unit' as const},
      expected: [8, 8]
    }
  ]) {
    for (const Operation of [
      GPUSpatialWeightsValidationOperation,
      CPUSpatialWeightsValidationOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.from(scene.neighbors)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)}
        },
        operations: [
          new Operation({
            ...scene.options,
            rowCount: 2,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {flags: 'flags', status: {invalidCount: 'invalid'}}
          })
        ],
        outputs: ['flags', 'invalid']
      });
      expect(Array.from(result['flags'])).toEqual(scene.expected);
      expect(Array.from(result['invalid'])).toEqual([
        scene.expected.filter(word => word !== 0).length
      ]);
    }
  }
});

it('UST-11 whole and chunked validators use global diagonal rows and declared columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of [
    {
      neighbors: [0, 1],
      values: [1, 1],
      options: {diagonal: 'require-unit' as const},
      expected: [0, 0]
    },
    {
      neighbors: [0, 1],
      values: [0.5, 1],
      options: {diagonal: 'require-unit' as const},
      expected: [8, 0]
    },
    {neighbors: [2, 0], values: [1, 1], options: {columnCount: 3}, expected: [0, 0]},
    {neighbors: [2, 0], values: [1, 1], options: {}, expected: [2, 0]}
  ]) {
    for (const chunked of [false, true]) {
      const harness = createRasterTimeSeriesHarness(device);
      const output = {
        flags: harness.output('flags', 'uint32', 2),
        status: {invalidCount: harness.output('invalid', 'uint32', 1)}
      };
      const producer = chunked
        ? new GPUChunkedSpatialWeightsValidation({
            ...scene.options,
            output,
            weights: {
              rowCount: 2,
              chunks: scene.neighbors.map((neighbor, rowStart) => ({
                rowStart,
                rowCount: 1,
                weights: {
                  offsets: harness.integers([0, 1]),
                  neighbors: harness.integers([neighbor]),
                  weights: harness.floats([scene.values[rowStart]])
                }
              }))
            }
          })
        : new GPUSpatialWeightsValidation({
            ...scene.options,
            output,
            weights: {
              offsets: harness.integers([0, 1, 2]),
              neighbors: harness.integers(scene.neighbors),
              weights: harness.floats(scene.values)
            }
          });
      const result = await harness.run(producer);
      expect(result['flags']).toEqual(scene.expected);
      expect(result['invalid']).toEqual([scene.expected.filter(word => word !== 0).length]);
    }
  }
});

it('the CPU operation accepts zero neighbor capacity while the GPU contributor rejects it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 0)},
    neighbors: {format: 'uint32' as const, values: new Uint32Array()},
    weights: {format: 'float32' as const, values: new Float32Array()}
  };
  const props = {
    id: 'empty-validation',
    rowCount: 1,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    output: {flags: 'flags', status: {invalidCount: 'invalid'}}
  };
  await expect(
    runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialWeightsValidationOperation(props)],
      outputs: ['flags', 'invalid']
    })
  ).rejects.toThrow(/empty-validation weights.neighbors length/);
  const result = await runOperationGraph(device, {
    inputs,
    operations: [new CPUSpatialWeightsValidationOperation(props)],
    outputs: ['flags', 'invalid']
  });
  expect(Array.from(result['flags'])).toEqual([0]);
  expect(Array.from(result['invalid'])).toEqual([0]);
});

it('compares every status and flags word on valid, invalid and empty-neighborhood CSR', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenes = [
    {
      offsets: Uint32Array.of(0, 2, 3, 4),
      neighbors: Uint32Array.of(1, 2, 0, 0, 0xffffffff),
      weights: Float32Array.of(0.5, 0.5, 1, 1, NaN),
      expectedFlags: [0, 0, 0],
      expectedInvalidCount: 0
    },
    {
      offsets: Uint32Array.of(0, 3, 4, 5, 6),
      neighbors: Uint32Array.of(0, 1, 1, 2, 9, 3),
      weights: Float32Array.of(1, 1, NaN, 0, -1, 1),
      expectedFlags: [28, 0, 18, 8],
      expectedInvalidCount: 3
    },
    {
      offsets: Uint32Array.of(1, 0, 9),
      neighbors: Uint32Array.of(0, 0),
      weights: Float32Array.of(Infinity, -1),
      expectedFlags: [1, 21],
      expectedInvalidCount: 2
    },
    {
      offsets: Uint32Array.of(0, 0, 0),
      neighbors: Uint32Array.of(0xffffffff),
      weights: Float32Array.of(NaN),
      expectedFlags: [0, 0],
      expectedInvalidCount: 0
    },
    {
      // Independent NumPy predicates: weights-validation-audit.oracle.py.
      offsets: Uint32Array.of(1, 3, 5, 7, 9, 9, 11, 10, 15),
      neighbors: Uint32Array.of(6, 0, 1, 4, 4, 9, 0, 3, 1, 6, 7, 0, 8, 7),
      weights: Float32Array.of(1, 1, NaN, 1, Infinity, -1, 1, 1, 1, 0, 2, 1, 1, NaN),
      expectedFlags: [25, 20, 22, 12, 0, 0, 1, 31],
      expectedInvalidCount: 6
    }
  ];
  for (const scene of scenes.flatMap(scene => [
    scene,
    {...scene, weights: Float32Array.from(scene.weights, value => value * 2)}
  ])) {
    for (const flags of ['flags', undefined]) {
      const props = {
        rowCount: scene.offsets.length - 1,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        output: {flags, status: {invalidCount: 'invalid'}}
      };
      const outputs = flags ? ['flags', 'invalid'] : ['invalid'];
      const results = [];
      for (const Operation of [
        GPUSpatialWeightsValidationOperation,
        CPUSpatialWeightsValidationOperation
      ])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              offsets: {format: 'uint32', values: scene.offsets},
              neighbors: {format: 'uint32', values: scene.neighbors},
              weights: {format: 'float32', values: scene.weights}
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      // Exact independent invariant words, not just GPU/CPU agreement. The
      // first four tiny fixtures are hand-assigned from the five published bits.
      for (const result of results) {
        expect(Array.from(result['invalid'])).toEqual([scene.expectedInvalidCount]);
        if (flags) expect(Array.from(result['flags'])).toEqual(scene.expectedFlags);
      }
    }
  }
});
