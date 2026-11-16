// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

/** A physical fixture buffer has one graph import, shared by every typed output subview. */
export function createCompactionTestViewImporter(
  graph: GPUCommandGraph,
  track: (buffer: Buffer) => Buffer,
  offsets: Readonly<Record<string, number>> = {},
  useSubviews = false
) {
  const handles = new Map<Buffer, GraphDataView['buffer']>();
  return <Format extends 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    let handle = handles.get(buffer);
    if (!handle) {
      handle = importGraphBuffer(graph, name, track(buffer), 'uint32').buffer;
      handles.set(buffer, handle);
    }
    return graph.createDataView(handle, {
      format,
      length,
      byteOffset: useSubviews ? (offsets[name] ?? 4) : 0
    });
  };
}
