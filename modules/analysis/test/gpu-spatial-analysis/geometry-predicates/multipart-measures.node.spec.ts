// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {ScalarData, WgslExec} from 'wgsl_reflect';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {
  GPUAffineTransform,
  getGPUAffineTransformParameterValues
} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createMultipartGeometry,
  FIXTURE,
  getMultipartMeasureLayout
} from './multipart-geometry-test-utils';

useCPUExecutorTestHooks();

// Local interpreter workarounds only; production WGSL and the headless cases are unchanged.
// wgsl_reflect lacks ldexp when a module does not also use frexp, and u32(1.5) retains an
// f32 1.5 internally (it writes 1 to storage but compares unequal to 1u).
beforeEach(() => {
  const originalBuiltin = WgslExec.prototype._callBuiltinFunction;
  vi.spyOn(WgslExec.prototype, '_callBuiltinFunction').mockImplementation(function (
    this: WgslExec,
    node,
    context
  ) {
    if (node.name === 'ldexp') {
      const value = this.evalExpression(node.args![0], context) as ScalarData;
      const power = this.evalExpression(node.args![1], context) as ScalarData;
      return new ScalarData(Math.fround(value.value * 2 ** power.value), this.getTypeInfo('f32')!);
    }
    return originalBuiltin.call(this, node, context);
  });
  const originalCreate = WgslExec.prototype._evalCreate;
  vi.spyOn(WgslExec.prototype, '_evalCreate').mockImplementation(function (
    this: WgslExec,
    node,
    context
  ) {
    const result = originalCreate.call(this, node, context);
    const name = 'type' in node ? node.type?.name : node.name;
    if (result instanceof ScalarData && (name === 'u32' || name === 'i32')) {
      return new ScalarData(
        name === 'u32' ? result.value >>> 0 : result.value | 0,
        this.getTypeInfo(name)!
      );
    }
    return result;
  });
});
afterEach(() => vi.restoreAllMocks());

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} measures and affine origins use complete features against Shapely`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const records = FIXTURE[kind];
    const geometry = createMultipartGeometry(
      kind,
      records.map(record => record.left),
      (name, values, format) => {
        const view = createTransientView(
          graph,
          name,
          format,
          format === 'float32x2' ? values.length / 2 : values.length
        );
        memory.write(view, values);
        return view;
      }
    );
    const lengths = createTransientView(graph, 'lengths', 'float32', records.length);
    const areas = createTransientView(graph, 'areas', 'float32', records.length);
    memory.write(areas, new Float32Array(records.length).fill(12345));
    const centroids = createTransientView(graph, 'centroids', 'float32x2', records.length);
    const bounds = createTransientView(graph, 'bounds', 'float32x4', records.length);
    const vertexCounts = createTransientView(graph, 'vertexCounts', 'uint32', records.length);
    const extremeVertices = createTransientView(
      graph,
      'extremeVertices',
      'uint32x4',
      records.length
    );
    const invalidCount = createTransientView(graph, 'invalidCount', 'uint32', 1);
    const output = {
      lengths,
      areas,
      centroids,
      bounds,
      vertexCounts,
      extremeVertices,
      status: {invalidCount}
    };
    memory.write(lengths, new Float32Array(records.length).fill(Number.NaN));
    memory.write(invalidCount, Uint32Array.of(99));
    const measures = new GPUGeometryMeasures({...getMultipartMeasureLayout(geometry), output});
    const transformed = createTransientView(
      graph,
      'transformed',
      'float32x2',
      geometry.positions.length
    );
    const parameters = createTransientView(graph, 'affineParameters', 'float32', 12);
    memory.write(
      parameters,
      getGPUAffineTransformParameterValues({scale: [2, -1], translate: [3, 4], origin: 'center'})
    );
    const affine = new GPUAffineTransform({
      ...getMultipartMeasureLayout(geometry),
      origins: ['center'],
      parameters,
      output: {positions: transformed}
    });
    try {
      executeCommandNodesOnCPU(
        [...measures.getCommandNodes(graph), ...affine.getCommandNodes(graph)],
        memory
      );
      const actualLengths = memory.readFloat32(lengths);
      expect(Array.from(memory.readFloat32(areas))).toEqual(records.map(record => record.area));
      const actualCentroids = memory.readFloat32(centroids);
      const actualBounds = memory.readFloat32(bounds);
      let vertexBase = 0;
      const expectedExtremes: number[] = [];
      for (const [row, record] of records.entries()) {
        expect(Number.isFinite(actualLengths[row]), record.name).toBe(true);
        // At most a few dozen f32 sums of coordinates <= 64: 2e-5 absolute allows rounding only.
        expect(Math.abs(actualLengths[row] - record.length), record.name).toBeLessThan(2e-5);
        for (const [values, expected, width] of [
          [actualCentroids, record.centroid, 2],
          [actualBounds, record.bounds, 4]
        ] as const) {
          for (let component = 0; component < width; component++) {
            const actual = values[row * width + component];
            if (expected === null) expect(Number.isNaN(actual), `${record.name} empty`).toBe(true);
            else {
              expect(Number.isFinite(actual), record.name).toBe(true);
              expect(Math.abs(actual - expected[component]), record.name).toBeLessThan(2e-5);
            }
          }
        }
        expectedExtremes.push(
          ...record.extremeVertices.map(vertex =>
            vertex === 0xffffffff ? vertex : vertex + vertexBase
          )
        );
        vertexBase += record.expected.numCoordinates;
      }
      expect(Array.from(memory.readUint32(vertexCounts))).toEqual(
        records.map(record => record.expected.numCoordinates)
      );
      expect(Array.from(memory.readUint32(extremeVertices))).toEqual(expectedExtremes);
      expect(Array.from(memory.readUint32(invalidCount))).toEqual([0]);
      const actualTransformed = memory.readFloat32(transformed);
      records
        .flatMap(record => record.affineCenter.flat())
        .forEach((expected, index) => {
          expect(Number.isFinite(actualTransformed[index])).toBe(true);
          expect(
            Math.abs(actualTransformed[index] - expected),
            `coordinate ${index}: got ${actualTransformed[index]}, Shapely ${expected}`
          ).toBeLessThan(2e-5);
        });
    } finally {
      device.destroy();
    }
  });
}
