// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUPolygonTriangulationOperation} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {GPUPolygonTriangulationOperation} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  OPERATION_POLYGONS,
  CLAMPED_RING_END_CASES,
  createClampedRingEndInput,
  getTriangulationOperationProps,
  packOperationPolygons
} from './polygon-triangulation-operation-cases';

for (const scenario of ['ordinary', 'empty-and-overflow', 'closed-ring'] as const) {
  it(`polygon operations compare every output exactly on ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = packOperationPolygons(
      scenario === 'ordinary'
        ? OPERATION_POLYGONS
        : scenario === 'empty-and-overflow'
          ? [
              [],
              [
                [
                  [0, 0],
                  [1, 0],
                  [2, 0]
                ]
              ],
              [
                [
                  [10, 0],
                  [14, 0],
                  [14, 4],
                  [10, 4]
                ]
              ]
            ]
          : [
              [
                [
                  [0, 0],
                  [4, 0],
                  [4, 4],
                  [0, 4],
                  [0, 0]
                ]
              ]
            ]
    );
    const props = getTriangulationOperationProps(
      input,
      scenario === 'empty-and-overflow' ? 3 : undefined
    );
    const inputs: Record<string, OperationGraphInput> = {
      positions: {format: 'float32x2', values: input.positions},
      featureOffsets: {format: 'uint32', values: input.featureOffsets},
      polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
      ringOffsets: {format: 'uint32', values: input.ringOffsets}
    };
    const outputs = ['indices', 'valid', 'triangleCount', 'triangleOffsets'];
    const results = [];
    for (const Operation of [GPUPolygonTriangulationOperation, CPUPolygonTriangulationOperation]) {
      // runOperationGraph owns/destroys the operation; each call receives a fresh instance.
      results.push(
        await runOperationGraph(device, {inputs, operations: [new Operation(props)], outputs})
      );
    }
    // Tolerance zero: these outputs are indices, counts and flags. Zero signs are compared
    // numerically, never by Object.is. Ordinary polygons exercise holes, concavity and a bridge
    // into an edge, all below the default threshold for GPU monotone routing.
    for (const output of outputs) {
      expect(results[0][output].length).toBe(results[1][output].length);
      for (let word = 0; word < results[0][output].length; word++) {
        expect(results[0][output][word] === results[1][output][word], `${output}[${word}]`).toBe(
          true
        );
      }
    }
    expect(Array.from(results[0]['valid'])).toEqual(
      scenario === 'empty-and-overflow' ? [1, 1, 0] : scenario === 'ordinary' ? [1, 1, 1] : [1]
    );
  });
}

for (const scene of CLAMPED_RING_END_CASES) {
  it(`polygon operations preserve wrapped failure padding for ring end ${scene.ringEnd}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createClampedRingEndInput(scene.ringEnd);
    const props = {
      ...getTriangulationOperationProps(input, 6),
      largePolygonMethod: 'none' as const
    };
    const outputs = ['indices', 'valid', 'triangleCount', 'triangleOffsets'];
    for (const Operation of [GPUPolygonTriangulationOperation, CPUPolygonTriangulationOperation]) {
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
      // Kernel-derived GPU hypothesis, pending device execution; every integer word is exact.
      expect(Array.from(result['indices'])).toEqual(scene.indices);
      expect(Array.from(result['valid'])).toEqual([0]);
      expect(Array.from(result['triangleCount'])).toEqual([0]);
      expect(Array.from(result['triangleOffsets'])).toEqual([0, scene.triangleEnd]);
    }
  });
}
