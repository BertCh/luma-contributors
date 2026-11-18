// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createGPUMultipartRowOwnersNode} from '../../../src/analysis-core/contracts/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUAffineTransform} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} owners agree with measures and affine feature rows on reuse`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const track = <BufferType extends Buffer>(buffer: BufferType): BufferType => {
      buffers.push(buffer);
      return buffer;
    };
    const coordinates =
      kind === 'points'
        ? Float32Array.of(0, 0, 2, 0, 10, 0, 12, 0)
        : Float32Array.of(0, 0, 2, 0, 10, 0, 12, 0, 20, 0, 23, 0);
    const partCount = 4;
    const featureCount = 5;
    const positions = importGraphBuffer(
      graph,
      'positions',
      track(createInputBuffer(device, coordinates)),
      'float32x2',
      coordinates.length / 2
    );
    const offsetBuffer = track(createInputBuffer(device, Uint32Array.of(0, 0, 2, 2, 4, 4)));
    const featureOffsets = importGraphBuffer(
      graph,
      'features',
      offsetBuffer,
      'uint32',
      featureCount + 1
    );
    const lineOffsets = importGraphBuffer(
      graph,
      'parts',
      track(createInputBuffer(device, Uint32Array.of(0, 2, 2, 4, 6))),
      'uint32',
      partCount + 1
    );
    const sourceIds = importGraphBuffer(
      graph,
      'identities',
      track(createInputBuffer(device, Uint32Array.of(99, 99, 99, 99, 99))),
      'uint32',
      featureCount
    );
    const ownerBuffer = track(createOutputBuffer(device, partCount * 4));
    const featureRows = importGraphBuffer(graph, 'owners', ownerBuffer, 'uint32', partCount);
    const countBuffer = track(createOutputBuffer(device, featureCount * 4));
    const vertexCounts = importGraphBuffer(graph, 'counts', countBuffer, 'uint32', featureCount);
    const transformedBuffer = track(createOutputBuffer(device, coordinates.byteLength));
    const transformed = importGraphBuffer(
      graph,
      'transformed',
      transformedBuffer,
      'float32x2',
      positions.length
    );
    // Distinct translations for feature rows 0, 1 and 3 expose any part-row or caller-ID lookup.
    const transforms = importGraphBuffer(
      graph,
      'transforms',
      track(
        createInputBuffer(
          device,
          Float32Array.of(
            1,
            0,
            0,
            1,
            10,
            0,
            1,
            0,
            0,
            1,
            100,
            0,
            1,
            0,
            0,
            1,
            500,
            0,
            1,
            0,
            0,
            1,
            1000,
            0,
            1,
            0,
            0,
            1,
            5000,
            0
          )
        )
      ),
      'float32',
      featureCount * 6
    );
    const geometry = {kind, positions, lineOffsets, featureOffsets, sourceIds};
    graph.add(createGPUMultipartRowOwnersNode(graph, 'owners-node', geometry, featureRows));
    graph.add(
      new GPUGeometryMeasures({
        ...geometry,
        geometryType: kind,
        output: {vertexCounts}
      })
    );
    graph.add(
      new GPUAffineTransform({
        ...geometry,
        geometryType: kind,
        featureTransforms: transforms,
        output: {positions: transformed}
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(ownerBuffer, partCount))).toEqual([1, 1, 3, 3]);
      expect(Array.from(await readUint32(countBuffer, featureCount))).toEqual(
        kind === 'points' ? [0, 2, 0, 2, 0] : [0, 2, 0, 4, 0]
      );
      const expected =
        kind === 'points'
          ? [100, 0, 102, 0, 1010, 0, 1012, 0]
          : [100, 0, 102, 0, 1010, 0, 1012, 0, 1020, 0, 1023, 0];
      expect(Array.from(await readFloat32(transformedBuffer, coordinates.length))).toEqual(
        expected
      );
      offsetBuffer.write(Uint32Array.of(0, 1, 1, 1, 4, 4));
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(ownerBuffer, partCount))).toEqual([0, 3, 3, 3]);
      expect(Array.from(await readUint32(countBuffer, featureCount))).toEqual(
        kind === 'points' ? [1, 0, 0, 3, 0] : [2, 0, 0, 4, 0]
      );
      const updated =
        kind === 'points'
          ? [10, 0, 1002, 0, 1010, 0, 1012, 0]
          : [10, 0, 12, 0, 1010, 0, 1012, 0, 1020, 0, 1023, 0];
      expect(Array.from(await readFloat32(transformedBuffer, coordinates.length))).toEqual(updated);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
