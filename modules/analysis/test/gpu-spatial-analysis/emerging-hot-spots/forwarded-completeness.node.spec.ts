// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {createInputBuffer, createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const certificateValue of [0, 1]) {
  for (const location of ['matrix', 'outer', 'chunk'] as const) {
    it(`N1 hot spots refuse ${location} complete=${certificateValue} before snapshot evidence is lost`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const buffer = createInputBuffer(device, Uint32Array.of(certificateValue));
      try {
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32' | 'sint32'>(
          format: Format,
          length: number
        ) => createTransientView(graph, `certificate-${serial++}`, format, length);
        const complete = importGraphBuffer(graph, 'complete', buffer, 'uint32', 1);
        const status = {complete};
        const matrix = {
          offsets: view('uint32', 3),
          neighbors: view('uint32', 2),
          weights: view('float32', 2),
          ...(location === 'matrix' || location === 'chunk' ? {status} : {})
        };
        const weights =
          location === 'matrix'
            ? matrix
            : {
                rowCount: 2,
                chunks: [{rowStart: 0, rowCount: 2, weights: matrix}],
                ...(location === 'outer' ? {status} : {})
              };
        expect(
          () =>
            new GPUEmergingHotSpots({
              weights,
              values: view('float32', 4),
              parameters: view('float32', 8),
              sliceCount: 2,
              multipleTesting: 'none',
              output: {
                giZScores: view('float32', 4),
                trendZ: view('float32', 2),
                trendP: view('float32', 2),
                trendS: view('sint32', 2),
                category: view('uint32', 2),
                hotSliceCount: view('uint32', 2),
                coldSliceCount: view('uint32', 2)
              }
            })
        ).toThrow(/weights.status.complete requires a completeness-aware consumer/);
        expect(matrix.status?.complete ?? status.complete).toBe(complete);
      } finally {
        buffer.destroy();
        device.destroy();
      }
    });
  }
}
