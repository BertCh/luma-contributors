// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';

/** Writes every pair slot's histogram membership on every submission, including inactive slots. */
export function addPairHistogramMask(
  graph: GPUCommandGraph<undefined>,
  pairs: {rightRows: GraphDataView<'uint32'>; status: {count: GraphDataView<'uint32'>}},
  id: string
): GraphDataView<'uint32'> {
  const mask = createTransientView(graph, id, 'uint32', pairs.rightRows.length, Buffer.STORAGE);
  graph.add(
    createWGSLKernelNode(graph, {
      id,
      operation: 'lens-pair-histogram-membership',
      invocationCount: mask.length,
      bindings: [
        {name: 'pairCount', view: pairs.status.count, type: 'u32', access: 'read'},
        {name: 'mask', view: mask, type: 'u32', access: 'read_write'}
      ],
      body: 'mask[maskOffset + index] = select(0u, 1u, index < pairCount[pairCountOffset]);'
    })
  );
  return mask;
}
