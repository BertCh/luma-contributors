// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPURegionMask,
  GPURegionStatistics,
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength
} from '../../../src/gpu-spatial-analysis/region-statistics/index';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('ENT-02 consumes foreign-constructor-shaped chunks without dropping either row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const port = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createOutputBuffer(device, values.length);
    buffer.write(values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `entry-${serial++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const vector = <Format extends 'float32' | 'float32x2'>(
    format: Format,
    data: readonly Float32Array[]
  ) => {
    const components = format === 'float32' ? 1 : 2;
    const local = new GraphVectorView({
      id: `entry-vector-${serial++}`,
      name: format,
      format,
      length: 2,
      valueLength: 2 * components,
      stride: components,
      byteStride: 4 * components,
      rowByteLength: 4 * components,
      data: data.map(values => port(format, values).view)
    });
    // The Node companion uses actual CommonJS imports. In the browser, preserve that public
    // shape while using another constructor identity; chunk/GPU buffer identities stay intact.
    return Object.assign(Object.create(Object.prototype), local) as typeof local;
  };
  const positions = vector('float32x2', [Float32Array.of(0, 0), Float32Array.of(1, 0)]);
  const values = vector('float32', [Float32Array.of(2), Float32Array.of(4)]);
  const bounds = port('float32', Float32Array.of(-1, -1, 2, 1)).view;
  const mask = port('uint32', Uint32Array.of(99, 99));
  const summary = port('uint32', new Uint32Array(getGPURegionStatisticsSummaryLength()).fill(99));
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      new GPURegionMask({
        id: 'entry-mask',
        positions,
        region: {kind: 'rectangle', bounds},
        output: {mask: mask.view, status: {overflow: port('uint32', Uint32Array.of(99)).view}}
      })
    );
    graph.add(
      new GPURegionStatistics({
        id: 'entry-stats',
        positions,
        values,
        region: {kind: 'rectangle', bounds},
        output: {summary: summary.view}
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect(await readUint32(mask.buffer)).toEqual([1, 1]);
    const result = decodeGPURegionStatistics(await summary.buffer.readAsync());
    // Two selected literal observations: 2+4=6, mean 3, extrema 2 and 4.
    expect(result).toMatchObject({
      selectedCount: 2,
      valueCount: 2,
      sum: 6,
      mean: 3,
      minimum: 2,
      maximum: 4,
      selectionTruncated: false,
      regionTruncated: false
    });
    for (const value of [result.sum, result.mean, result.minimum, result.maximum])
      expect(Number.isFinite(value)).toBe(true);
  } finally {
    compiled?.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
