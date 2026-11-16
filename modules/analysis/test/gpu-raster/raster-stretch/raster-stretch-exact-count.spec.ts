// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterStretch} from '../../../src/raster-analysis/raster-stretch/gpu-raster-stretch';
import {getGPURasterStretchParameterValues} from '../../../src/raster-analysis/raster-stretch/raster-stretch-parameters';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('reports an exact odd finite-cell count beyond 2^24 while retaining the legacy float statistics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 4097;
  const length = width * width;
  const samples = new Float32Array(length).fill(1);
  samples[0] = NaN;
  samples[1] = NaN;
  const values = createInputBuffer(device, samples);
  const parameters = createInputBuffer(device, getGPURasterStretchParameterValues());
  const validCount = createOutputBuffer(device, 1);
  const statistics = createOutputBuffer(device, 8);
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPURasterStretch({
      grid: {width, height: width, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      values: importGraphBuffer(graph, 'values', values, 'float32', length),
      parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 16),
      binCount: 16,
      output: {
        validCount: importGraphBuffer(graph, 'count', validCount, 'uint32', 1),
        statistics: importGraphBuffer(graph, 'statistics', statistics, 'float32', 8)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(validCount, 1)).toEqual([length - 2]);
    const summary = await readFloat32(statistics, 8);
    expect(summary[0]).toBe(1);
    expect(summary[1]).toBe(1);
    expect(summary[4]).toBe(Math.fround(length - 2));
  } finally {
    compiled.destroy();
    for (const buffer of [values, parameters, validCount, statistics]) buffer.destroy();
  }
}, 120000);

it('keeps partial workgroups uniform and bins large or tiny normal scales using an independent oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 257;
  for (const binCount of [16, 65536]) {
    for (const [minimum, maximum] of [
      [0, 2 ** 127],
      [0, 2 ** -125],
      [-(2 ** 127), 2 ** 127]
    ] as const) {
      const samples = Float32Array.from(
        {length},
        (_, index) => minimum + ((index % 3) * (maximum - minimum)) / 2
      );
      const values = createInputBuffer(device, samples);
      const parameters = createInputBuffer(
        device,
        // The automatic domain's opposite extremes have a span larger than finite float32.
        getGPURasterStretchParameterValues(minimum < 0 ? {} : {domain: [minimum, maximum]})
      );
      const histogram = createOutputBuffer(device, binCount);
      const validCount = createOutputBuffer(device, 1);
      const stretched = createOutputBuffer(device, length);
      const graph = new GPUCommandGraph(device);
      graph.add(
        new GPURasterStretch({
          grid: {width: length, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          values: importGraphBuffer(graph, 'values', values, 'float32', length),
          parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 16),
          binCount,
          output: {
            histogram: importGraphBuffer(graph, 'histogram', histogram, 'uint32', binCount),
            validCount: importGraphBuffer(graph, 'count', validCount, 'uint32', 1),
            stretched: importGraphBuffer(graph, 'stretched', stretched, 'float32', length)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const expectedHistogram = new Array<number>(binCount).fill(0);
        for (let index = 0; index < length; index++) {
          const bin = index % 3 === 0 ? 0 : index % 3 === 1 ? binCount / 2 : binCount - 1;
          expectedHistogram[bin]++;
        }
        expect(await readUint32(validCount, 1)).toEqual([length]);
        expect(await readUint32(histogram, binCount)).toEqual(expectedHistogram);
        const result = await readFloat32(stretched, length);
        for (let index = 0; index < length; index++)
          expect(result[index]).toBeCloseTo((index % 3) / 2, 6);
      } finally {
        compiled.destroy();
        for (const buffer of [values, parameters, histogram, validCount, stretched])
          buffer.destroy();
      }
    }
  }
});
