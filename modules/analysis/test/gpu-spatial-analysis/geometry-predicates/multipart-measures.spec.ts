// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {
  GPUAffineTransform,
  getGPUAffineTransformParameterValues
} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createMultipartGeometry,
  FIXTURE,
  getMultipartMeasureLayout
} from './multipart-geometry-test-utils';

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} measures and affine centers match Shapely for complete features`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const records = FIXTURE[kind];
    const geometry = createMultipartGeometry(
      kind,
      records.map(record => record.left),
      (name, values, format) => {
        const buffer = createInputBuffer(device, values);
        buffers.push(buffer);
        return importGraphBuffer(
          graph,
          name,
          buffer,
          format,
          format === 'float32x2' ? values.length / 2 : values.length
        );
      }
    );
    const output = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x4'>(
      name: string,
      format: Format,
      length = records.length
    ) => {
      const width = format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;
      const buffer = createOutputBuffer(device, length * width);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const lengths = output('lengths', 'float32');
    const areas = output('areas', 'float32');
    areas.buffer.write(new Float32Array(records.length).fill(12345));
    const centroids = output('centroids', 'float32x2');
    const bounds = output('bounds', 'float32x4');
    const vertexCounts = output('vertexCounts', 'uint32');
    const extremeVertices = output('extremeVertices', 'uint32x4');
    const invalidCount = output('invalidCount', 'uint32', 1);
    lengths.buffer.write(new Float32Array(records.length).fill(Number.NaN));
    invalidCount.buffer.write(Uint32Array.of(99));
    const transformed = output('transformed', 'float32x2', geometry.positions.length);
    const parameterBuffer = createInputBuffer(
      device,
      getGPUAffineTransformParameterValues({scale: [2, -1], translate: [3, 4], origin: 'center'})
    );
    buffers.push(parameterBuffer);
    const parameters = importGraphBuffer(graph, 'affineParameters', parameterBuffer, 'float32', 12);
    graph.add(
      new GPUGeometryMeasures({
        ...getMultipartMeasureLayout(geometry),
        output: {
          lengths: lengths.view,
          areas: areas.view,
          centroids: centroids.view,
          bounds: bounds.view,
          vertexCounts: vertexCounts.view,
          extremeVertices: extremeVertices.view,
          status: {invalidCount: invalidCount.view}
        }
      })
    );
    graph.add(
      new GPUAffineTransform({
        ...getMultipartMeasureLayout(geometry),
        origins: ['center'],
        parameters,
        output: {positions: transformed.view}
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actualLengths = await readFloat32(lengths.buffer, records.length);
      expect(await readFloat32(areas.buffer, records.length)).toEqual(
        records.map(record => record.area)
      );
      const actualCentroids = await readFloat32(centroids.buffer, records.length * 2);
      const actualBounds = await readFloat32(bounds.buffer, records.length * 4);
      const expectedExtremes: number[] = [];
      let vertexBase = 0;
      for (const [row, record] of records.entries()) {
        expect(Number.isFinite(actualLengths[row]), record.name).toBe(true);
        // <= 64 coordinates, a few dozen f32 sums: 2e-5 absolute covers rounding.
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
      expect(await readUint32(vertexCounts.buffer, records.length)).toEqual(
        records.map(record => record.expected.numCoordinates)
      );
      expect(await readUint32(extremeVertices.buffer, records.length * 4)).toEqual(
        expectedExtremes
      );
      expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
      const actualTransformed = await readFloat32(
        transformed.buffer,
        geometry.positions.length * 2
      );
      records
        .flatMap(record => record.affineCenter.flat())
        .forEach((expected, index) => {
          expect(Number.isFinite(actualTransformed[index])).toBe(true);
          expect(Math.abs(actualTransformed[index] - expected)).toBeLessThan(2e-5);
        });
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
