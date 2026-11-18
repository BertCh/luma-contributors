// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';

/** Uploads offset chunks used by the packed/chunked flow parity fixture. */
export function createOffsetFlowChunkViews<Format extends 'float32' | 'float32x2', Parameters>(
  device: Device,
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    format: Format;
    data: ArrayLike<number>;
    lengths: readonly number[];
    buffers: Buffer[];
  }
): GraphDataView<Format>[] {
  const {id, format, data, lengths, buffers} = props;
  const components = format === 'float32x2' ? 2 : 1;
  let rowBase = 0;
  return lengths.map((length, index) => {
    const padding = 70 + index;
    // An empty view still binds one physical row at its offset. Keep its logical length zero.
    const values = new Float32Array((padding + Math.max(length, 1)) * components).fill(999);
    for (let component = 0; component < length * components; component++) {
      values[padding * components + component] = data[rowBase * components + component];
    }
    rowBase += length;
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const backing = importGraphBuffer(graph, `${id}-${index}`, buffer, format, padding + length);
    return graph.createDataView(backing.buffer, {
      format,
      byteOffset: padding * components * Float32Array.BYTES_PER_ELEMENT,
      length
    });
  });
}
