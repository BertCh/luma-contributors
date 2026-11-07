// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
export {addBreaksSelectPass} from '../../../../spatial-analysis-shared/breaks-select-pass';

/**
 * Returns an importer that binds each buffer once per graph. A graph rejects two imports that
 * resolve to one physical buffer, so every node that reads or writes a buffer shares one view.
 */
export function createGraphImporter(graph: GPUCommandGraph<void>) {
  const floatViews = new Map<Buffer, GraphDataView<'float32'>>();
  const wordViews = new Map<Buffer, GraphDataView<'uint32'>>();
  return {
    float: (buffer: Buffer, length: number) => {
      let view = floatViews.get(buffer);
      if (!view) {
        view = importGraphBuffer(graph, buffer.id, buffer, 'float32', length);
        floatViews.set(buffer, view);
      }
      return view;
    },
    word: (buffer: Buffer, length: number) => {
      let view = wordViews.get(buffer);
      if (!view) {
        view = importGraphBuffer(graph, buffer.id, buffer, 'uint32', length);
        wordViews.set(buffer, view);
      }
      return view;
    }
  };
}
