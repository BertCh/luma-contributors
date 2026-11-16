// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUClassAssignment} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUClassAssignment matches packed and independent expected classes for uneven chunks and both closures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const noClass = 0xffffffff;
  for (const intervalClosure of ['left', 'right'] as const) {
    const expected =
      intervalClosure === 'left'
        ? [0, 0, 0, 1, 2, 2, 2, noClass, noClass]
        : [0, 0, 0, 0, 1, 2, 2, noClass, noClass];
    const results: number[][] = [];
    for (const lengths of [[9], [1, 3, 5]]) {
      const graph = new GPUCommandGraph(device);
      const valuesBuffer = createInputBuffer(
        device,
        Float32Array.of(-Infinity, -1, 0, 1, 2, 3, Infinity, NaN, 2)
      );
      const maskBuffer = createInputBuffer(device, Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 1, 0));
      const breaksBuffer = createInputBuffer(device, Float32Array.of(0, 1, 2, 3));
      const countBuffer = createInputBuffer(device, Uint32Array.of(3));
      const outputBuffer = createOutputBuffer(device, 9);
      const split = <Format extends 'float32' | 'uint32'>(view: GraphDataView<Format>) => {
        if (lengths.length === 1) return view;
        let rowBase = 0;
        return createVectorView(
          view.buffer.id,
          view.format,
          lengths.map(length => {
            const chunk = graph.createDataView(view.buffer, {
              format: view.format,
              byteOffset: rowBase * 4,
              length
            });
            rowBase += length;
            return chunk;
          })
        );
      };
      graph.add(
        new GPUClassAssignment({
          values: split(importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 9)),
          mask: split(importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 9)),
          output: split(importGraphBuffer(graph, 'output', outputBuffer, 'uint32', 9)),
          breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', 4),
          classCount: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          intervalClosure
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readUint32(outputBuffer, 9);
        expect(actual).toEqual(expected);
        results.push(actual);
        countBuffer.write(Uint32Array.of(0));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(outputBuffer, 9)).toEqual(Array<number>(9).fill(noClass));
      } finally {
        compiled.destroy();
        for (const buffer of [valuesBuffer, maskBuffer, breaksBuffer, countBuffer, outputBuffer])
          buffer.destroy();
      }
    }
    expect(results[1]).toEqual(results[0]);
  }
});
