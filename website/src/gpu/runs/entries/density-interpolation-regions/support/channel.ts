// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Operations such as GPULocalKriging write float32x4 rows ([prediction, variance, ...]); the demo
// layers read scalar buffers. This adds a one-line compute node to the same graph that copies one
// channel into a scalar buffer, so the layers still draw what the operation wrote.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

/** Adds a node copying `channel` of each float32x4 (or float32x2) row into a new float32 buffer. */
export function addChannel(
  builder: RunBuilder,
  source: RunBuffer<'float32x4' | 'float32x2'>,
  channel: number,
  id: string,
  componentCount = 4
): RunBuffer<'float32'> {
  const target = builder.allocate(id, 'float32', source.length);
  builder.graph.add({
    getCommandNodes: graph => [
      createWGSLKernelNode(graph, {
        id,
        operation: 'channel',
        bindings: [
          {name: 'channelSource', view: source.view, type: 'f32', access: 'read'},
          {name: 'channelTarget', view: target.view, type: 'f32', access: 'read_write'}
        ],
        invocationCount: source.length,
        body: `channelTarget[channelTargetOffset + index] = channelSource[channelSourceOffset + index * ${componentCount}u + ${channel}u];`
      })
    ]
  });
  return target;
}
