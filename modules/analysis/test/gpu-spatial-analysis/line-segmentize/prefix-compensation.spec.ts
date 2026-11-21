// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createLinePrefixNode} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const rowCount of [64, 193, 257]) {
  it(`GEO-131 preserves compensation in the ${rowCount === 64 ? 'short' : `cooperative ${rowCount} rows`} line prefix`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const positions = new Float32Array(2 * rowCount);
    for (let row = 1; row < rowCount; row++) {
      positions[2 * row] = rowCount === 193 ? Math.min(row, 64) * 16 : 16777216;
      positions[2 * row + 1] =
        rowCount === 193
          ? Math.max(row - 64, 0) * 2 ** -20
          : rowCount === 64
            ? Math.min(row - 1, 2)
            : Math.min(Math.max(row - 64, 0), 128) / 64;
    }
    const positionBuffer = createInputBuffer(device, positions);
    const offsetBuffer = createInputBuffer(device, Uint32Array.of(0, rowCount));
    const measureBuffer = createOutputBuffer(device, rowCount);
    graph.add(
      createLinePrefixNode(graph, {
        id: 'prefix',
        operation: 'GPULineSegmentize',
        coordinateSystem: 'planar',
        radius: 1,
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rowCount),
        lineOffsets: importGraphBuffer(graph, 'offsets', offsetBuffer, 'uint32', 2),
        rowMeasures: importGraphBuffer(graph, 'measures', measureBuffer, 'float32', rowCount)
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(measureBuffer, rowCount);
      // The 64/257-row controls have length 2^24 then two units. The short route adds 1 twice; the
      // cooperative route adds two tiles of 64*(1/64), then a zero tile exposes the carry.
      // A plain f32 sum loses each 1. Both routes must retain the final representable +2.
      expect(actual.every(Number.isFinite)).toBe(true);
      expect(actual[0]).toBe(0);
      expect(actual).toHaveLength(rowCount);
      if (rowCount === 193) {
        // 64*16 + 128*2^-20 is exactly representable. Assert the moving endpoint;
        // a stationary vertex must not be needed to expose the carry's correction.
        expect(actual[64]).toBe(1024);
        expect(actual[192]).toBe(1024.0001220703125);
      } else {
        expect(actual[1]).toBe(16777216);
        expect(actual[rowCount - 1]).toBe(16777218);
        if (rowCount === 257) expect(actual[192]).toBe(16777218);
      }
    } finally {
      compiled.destroy();
      positionBuffer.destroy();
      offsetBuffer.destroy();
      measureBuffer.destroy();
    }
  });
}
