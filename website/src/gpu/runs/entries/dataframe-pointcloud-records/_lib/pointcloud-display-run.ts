// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RunBuilder, RunBuffer} from '../../../define-run';
import {createPointCloudDisplayCopyNode} from './pointcloud-display';

export function addPointCloudDisplayCopy(
  builder: RunBuilder,
  id: string,
  source: RunBuffer,
  status: RunBuffer,
  length: number,
  options: {
    componentStart?: number;
    components?: 1 | 2;
    invalidBits?: number;
    statusIndex?: number;
  } = {}
): RunBuffer {
  const output = builder.allocate(id, options.components === 2 ? 'float32x2' : 'float32', length);
  builder.graph.add(
    createPointCloudDisplayCopyNode(builder.graph, {
      id,
      source: source.view,
      status: status.view as GraphDataView<'uint32'>,
      output: output.view,
      ...options
    })
  );
  return output;
}
