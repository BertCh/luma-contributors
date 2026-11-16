// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUConstrainedDelaunayTriangulationOperation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation';
import {GPUConstrainedDelaunayTriangulationOperation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  OPERATION_POLYGONS,
  CLAMPED_RING_END_CASES,
  createClampedRingEndInput,
  packOperationPolygons,
  getTriangulationOperationProps
} from './polygon-triangulation-operation-cases';

for (const flipMethod of ['serial', 'parallel'] as const) {
  for (const scenario of ['flips-and-holes', 'empty-and-overflow'] as const) {
    it(`constrained triangulation ${flipMethod} compares every integer output on ${scenario}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = packOperationPolygons(
        scenario === 'flips-and-holes'
          ? [
              [
                [
                  [0, 0],
                  [4, 0],
                  [4, 4],
                  [0, 1]
                ]
              ],
              ...OPERATION_POLYGONS.map(rings =>
                rings.map(ring =>
                  ring.map(([horizontal, vertical]) => [horizontal + 100, vertical] as const)
                )
              )
            ]
          : [
              [],
              [
                [
                  [10, 0],
                  [14, 0],
                  [14, 4],
                  [10, 4]
                ]
              ],
              []
            ]
      );
      const base = getTriangulationOperationProps(
        input,
        scenario === 'empty-and-overflow' ? 3 : undefined
      );
      const props = {
        ...base,
        flipMethod,
        maximumFlips: 64,
        output: {...base.output, status: {converged: 'converged'}}
      };
      const outputs = ['indices', 'valid', 'triangleCount', 'triangleOffsets', 'converged'];
      const results = [];
      for (const Operation of [
        GPUConstrainedDelaunayTriangulationOperation,
        CPUConstrainedDelaunayTriangulationOperation
      ]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: input.positions},
              featureOffsets: {format: 'uint32', values: input.featureOffsets},
              polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
              ringOffsets: {format: 'uint32', values: input.ringOffsets}
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      }
      // Exact comparison: the bounded GPU completes these small slices in the same pair/side order.
      for (const output of outputs)
        expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
      expect(Array.from(results[0]['converged'])).toEqual(
        scenario === 'empty-and-overflow' ? [1, 0, 1] : [1, 1, 1, 1]
      );
      if (scenario === 'flips-and-holes')
        expect(Array.from(results[0]['indices'].subarray(0, 6))).toEqual([3, 1, 2, 1, 3, 0]);
    });
  }
}

for (const scene of CLAMPED_RING_END_CASES) {
  it(`constrained operations preserve wrapped failure padding for ring end ${scene.ringEnd}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createClampedRingEndInput(scene.ringEnd);
    const base = getTriangulationOperationProps(input, 6);
    const props = {
      ...base,
      largePolygonMethod: 'none' as const,
      output: {...base.output, status: {converged: 'converged'}}
    };
    const outputs = ['indices', 'valid', 'triangleCount', 'triangleOffsets', 'converged'];
    for (const Operation of [
      GPUConstrainedDelaunayTriangulationOperation,
      CPUConstrainedDelaunayTriangulationOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: input.positions},
          featureOffsets: {format: 'uint32', values: input.featureOffsets},
          polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
          ringOffsets: {format: 'uint32', values: input.ringOffsets}
        },
        operations: [new Operation(props)],
        outputs
      });
      // Kernel-derived GPU hypothesis, pending device execution. Invalid slices skip Lawson.
      expect(Array.from(result['indices'])).toEqual(scene.indices);
      expect(Array.from(result['valid'])).toEqual([0]);
      expect(Array.from(result['triangleCount'])).toEqual([0]);
      expect(Array.from(result['triangleOffsets'])).toEqual([0, scene.triangleEnd]);
      expect(Array.from(result['converged'])).toEqual([0]);
    }
  });
}

it('compiles the largest safe serial flip budget and returns the hand-computed diagonal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 1]
      ]
    ]
  ]);
  const base = getTriangulationOperationProps(input);
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: input.positions},
      featureOffsets: {format: 'uint32', values: input.featureOffsets},
      polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
      ringOffsets: {format: 'uint32', values: input.ringOffsets}
    },
    operations: [
      new GPUConstrainedDelaunayTriangulationOperation({
        ...base,
        flipMethod: 'serial',
        maximumFlips: 0xfffffffe,
        output: {...base.output, status: {converged: 'converged'}}
      })
    ],
    outputs: ['indices', 'valid', 'triangleCount', 'triangleOffsets', 'converged']
  });
  // The only legal Delaunay diagonal is 1--3; pair/side order fixes these triangle rows.
  expect(Array.from(result['indices'])).toEqual([3, 1, 2, 1, 3, 0]);
  expect(Array.from(result['valid'])).toEqual([1]);
  expect(Array.from(result['triangleCount'])).toEqual([2]);
  expect(Array.from(result['triangleOffsets'])).toEqual([0, 2]);
  expect(Array.from(result['converged'])).toEqual([1]);
});
