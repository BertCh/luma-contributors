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
  createPaddingTriangulationInput,
  CLAMPED_RING_END_CASES,
  createClampedRingEndInput,
  getTriangulationOperationProps,
  packOperationPolygons
} from './polygon-triangulation-operation-cases';

for (const Operation of [GPUPolygonTriangulationOperation, CPUPolygonTriangulationOperation]) {
  it(`CORE-08 ${Operation.name} keeps its original triangle contract after props mutation`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = packOperationPolygons([
      [
        [
          [0, 0],
          [1, 0],
          [0, 1]
        ]
      ]
    ]);
    const props = {
      ...getTriangulationOperationProps(input, 3),
      maximumWork: 100,
      spatialContext: {coordinateSpace: 'planar' as const, metric: 'native' as const}
    };
    const operation = new Operation(props);
    props.polygonCount = 2;
    props.indexCapacity = 6;
    props.maximumWork = 0;
    props.positions = 'changed-positions';
    props.output.indices = 'changed-indices';
    delete props.output.triangleCount;
    delete props.output.triangleOffsets;
    Reflect.set(props.spatialContext, 'coordinateSpace', 'longitude-latitude');
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: input.positions},
        featureOffsets: {format: 'uint32', values: input.featureOffsets},
        polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
        ringOffsets: {format: 'uint32', values: input.ringOffsets}
      },
      operations: [operation],
      outputs: ['indices', 'valid', 'triangleCount', 'triangleOffsets']
    });
    expect(Array.from(result['triangleCount'])).toEqual([1]);
    expect(Array.from(result['triangleOffsets'])).toEqual([0, 1]);
    expect(Array.from(result['valid'])).toEqual([1]);
    expect(Array.from(result['indices']).sort()).toEqual([0, 1, 2]);
  });
}

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

// Adjacent ring/line intervals share a prefix offset, so internal position gaps are not representable.
it('Earcut, monotone and CPU triangulation never include position padding in polygon geometry', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const outputs = ['indices', 'valid', 'triangleCount', 'triangleOffsets'];
  for (const method of ['cpu', 'none', 'monotone'] as const) {
    let reference: Record<string, number[]> | undefined;
    for (const value of [undefined, Number.NaN, 3e30, -3e30]) {
      const leading = value === undefined ? 0 : 2;
      const input = createPaddingTriangulationInput(leading, leading ? 3 : 0, value);
      const props = {
        ...getTriangulationOperationProps(input, 128),
        largePolygonMethod: method === 'cpu' ? ('none' as const) : method,
        monotoneMinimumVertices: 1
      };
      const Operation =
        method === 'cpu' ? CPUPolygonTriangulationOperation : GPUPolygonTriangulationOperation;
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
      expect(Array.from(result['valid'])).toEqual([1, 1, 1]);
      expect(Array.from(result['triangleCount'])).toEqual([0, 8, 2]);
      const usedWords = 3 * result['triangleOffsets'][3];
      const normalized = {
        indices: Array.from(result['indices'].slice(0, usedWords), vertex => vertex - leading),
        valid: Array.from(result['valid']),
        triangleCount: Array.from(result['triangleCount']),
        triangleOffsets: Array.from(result['triangleOffsets'])
      };
      if (reference) expect(normalized).toEqual(reference);
      else reference = normalized;
      // Independent hand areas and positive winding: no triangle may touch a garbage row.
      for (let polygon = 0; polygon < 3; polygon++) {
        let twiceArea = 0;
        for (
          let triangle = result['triangleOffsets'][polygon];
          triangle < result['triangleOffsets'][polygon + 1];
          triangle++
        ) {
          const vertices = Array.from(result['indices'].slice(3 * triangle, 3 * triangle + 3));
          expect(vertices.every(vertex => vertex >= leading && vertex < leading + 12)).toBe(true);
          const [first, second, third] = vertices.map(vertex => [
            input.positions[2 * vertex],
            input.positions[2 * vertex + 1]
          ]);
          const cross =
            (second[0] - first[0]) * (third[1] - first[1]) -
            (second[1] - first[1]) * (third[0] - first[0]);
          expect(cross).toBeGreaterThan(0);
          twiceArea += cross;
        }
        expect(twiceArea / 2).toBe([0, 12, 4][polygon]);
      }
    }
  }
}, 120000);
