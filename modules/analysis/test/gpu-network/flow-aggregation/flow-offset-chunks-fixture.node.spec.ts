// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validateGraphChunkBindingRanges} from '../../../src/analysis-core/kernels/graph-chunk-dispatch';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {createOffsetFlowChunkViews} from './flow-offset-chunks-fixture';

it.each([
  {format: 'float32x2' as const, lengths: [0, 1, 17, 0, 240, 264, 0]},
  {format: 'float32x2' as const, lengths: [16, 0, 240, 1, 0, 255, 10]},
  {format: 'float32' as const, lengths: [0, 3, 256, 0, 1, 262, 0]}
])('flow offset fixture preserves $format empty chunks with bindable backing rows ($lengths)', ({
  format,
  lengths
}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const components = format === 'float32x2' ? 2 : 1;
  try {
    const views = createOffsetFlowChunkViews(device, graph, {
      id: 'flow-offset-input',
      format,
      data: new Float32Array(522 * components),
      lengths,
      buffers
    });
    expect(views.map(view => view.length)).toEqual(lengths);
    expect(views.map(view => view.byteOffset)).toEqual(
      lengths.map((_, index) => (70 + index) * components * Float32Array.BYTES_PER_ELEMENT)
    );
    expect(() =>
      validateGraphChunkBindingRanges(graph, 'flow-offset-fixture', {
        input: createVectorView('flow-offset-input', format, views)
      })
    ).not.toThrow();
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
});
