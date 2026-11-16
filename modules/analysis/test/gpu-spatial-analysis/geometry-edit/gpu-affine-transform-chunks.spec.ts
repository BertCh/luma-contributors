// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUAffineTransform,
  getGPUAffineTransformParameterValues
} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const geometryType of ['global', 'lines', 'polygons', 'points'] as const) {
  const perFeature = geometryType !== 'global';
  it(`affine ${geometryType} matches packed rows and a closed form`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const source = new Float32Array([1, 2, -2, 3, 4, -1, 0, 0, 6, 2, 2, -3, -1, 4]);
    const input = (name: string, values: Float32Array | Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        name,
        buffer,
        values instanceof Float32Array ? 'float32' : 'uint32',
        values.length
      );
    };
    const settings = input(
      'settings',
      perFeature
        ? new Float32Array([2, 0, 0, 3, 4, -5, -1, 0, 0, 2, 10, 2])
        : getGPUAffineTransformParameterValues({scale: [2, 3], translate: [4, -5], origin: [1, -1]})
    ) as GraphDataView<'float32'>;
    const lineOffsets = input(
      'line-offsets',
      new Uint32Array([0, 3, 7])
    ) as GraphDataView<'uint32'>;
    const polygonOffsets = input(
      'polygon-offsets',
      new Uint32Array([0, 1, 2])
    ) as GraphDataView<'uint32'>;
    const featureOffsets = input(
      'feature-offsets',
      new Uint32Array([0, 1, 2])
    ) as GraphDataView<'uint32'>;
    const outputs: {buffer: Buffer; prefix: number; length: number}[][] = [];
    for (const [routeIndex, lengths] of [[7], [1, 4, 2]].entries()) {
      let rowBase = 0;
      const positions: GraphDataView<'float32x2'>[] = [];
      const transformed: GraphDataView<'float32x2'>[] = [];
      const routeOutputs: {buffer: Buffer; prefix: number; length: number}[] = [];
      for (const [chunkIndex, length] of lengths.entries()) {
        const prefix = 1 + chunkIndex;
        const padded = new Float32Array(prefix + 2 * length);
        padded.set(source.subarray(2 * rowBase, 2 * (rowBase + length)), prefix);
        const backing = input(`input-${routeIndex}-${chunkIndex}`, padded);
        positions.push(
          graph.createDataView(backing.buffer, {
            format: 'float32x2',
            length,
            byteOffset: prefix * 4
          })
        );
        const outputPrefix = chunkIndex + 2;
        const buffer = createOutputBuffer(device, outputPrefix + 2 * length);
        buffers.push(buffer);
        const outputBacking = importGraphBuffer(
          graph,
          `output-${routeIndex}-${chunkIndex}`,
          buffer,
          'float32',
          outputPrefix + 2 * length
        );
        transformed.push(
          graph.createDataView(outputBacking.buffer, {
            format: 'float32x2',
            length,
            byteOffset: outputPrefix * 4
          })
        );
        routeOutputs.push({buffer, prefix: outputPrefix, length});
        rowBase += length;
      }
      graph.add(
        new GPUAffineTransform({
          id: `affine-${routeIndex}`,
          positions:
            routeIndex === 0
              ? positions[0]
              : createVectorView('chunked-input', 'float32x2', positions),
          ...(perFeature
            ? {
                featureTransforms: settings,
                ...(geometryType === 'polygons'
                  ? {
                      geometryType: 'polygons' as const,
                      ringOffsets: lineOffsets,
                      polygonOffsets,
                      featureOffsets
                    }
                  : geometryType === 'points'
                    ? {geometryType: 'points' as const, featureOffsets: lineOffsets}
                    : {lineOffsets})
              }
            : {parameters: settings}),
          output: {
            positions:
              routeIndex === 0
                ? transformed[0]
                : createVectorView('chunked-output', 'float32x2', transformed)
          }
        })
      );
      outputs.push(routeOutputs);
    }
    const compiled = graph.compile();
    try {
      // Encoding twice verifies parameter upload survives transient aliasing and graph reuse.
      for (let encoding = 0; encoding < 2; encoding++) {
        submitGraph(device, compiled, undefined);
        const actual = [];
        for (const route of outputs) {
          const values = [];
          for (const {buffer, prefix, length} of route)
            values.push(...(await readFloat32(buffer, prefix + length * 2)).slice(prefix));
          actual.push(values);
        }
        expect(actual[1]).toEqual(actual[0]);
        const expected = Array.from({length: 7}, (_, row) => {
          const x = source[2 * row];
          const y = source[2 * row + 1];
          return perFeature
            ? row < 3
              ? [2 * x + 4, 3 * y - 5]
              : [-x + 10, 2 * y + 2]
            : [2 * (x - 1) + 5, 3 * (y + 1) - 6];
        }).flat();
        expect(actual[0]).toEqual(expected);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
