// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// example:hide-start
// Abridged from modules/analysis/src/dataframe-analysis/time-window-filter/gpu-time-window-filter.ts
// (the float32 path without extra predicates) and time-window-classify-node.ts (the classify
// bindings). Only the WGSL text and the setup are hidden.
// example:hide-end

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUVisibilityWorkflow} from '@luma.gl/graph/gpu-core';
import {
  createPublishNode,
  createWGSLKernelNode,
  type GPUCompactOutput
} from '@luma-contributors/analysis/analysis-core';

// example:hide-start
declare const graph: GPUCommandGraph<void>;
declare const timeWindow: GraphDataView<'float32'>;
declare const timestamps: GraphDataView<'float32'>;
declare const timestampsLow: GraphDataView<'float32'>;
declare const outputMask: GraphDataView<'uint32'>;
declare const total: GraphDataView<'uint32'>;
declare const output: GPUCompactOutput;
declare const classifyDeclarations: string;
declare const classifyBody: string;
// example:hide-end
// GPUTimeWindowFilter, abridged: the kernels it adds, in the order it adds them.
const nodes = [
  // classify: one invocation per row writes 1 or 0 into outputMask.
  createWGSLKernelNode<void>(graph, {
    id: 'time-window-filter-classify',
    operation: 'GPUTimeWindowFilter',
    variant: 'classify',
    bindings: [
      {name: 'params', view: timeWindow, type: 'f32', access: 'read'},
      {name: 'startHigh', view: timestamps, type: 'f32', access: 'read'},
      {name: 'startLow', view: timestampsLow, type: 'f32', access: 'read'},
      {name: 'outputMask', view: outputMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: timestamps.length,
    declarations: classifyDeclarations,
    body: classifyBody
  }),
  // compaction (luma.gl): reads outputMask, scans it, scatters the kept row ids.
  ...new GPUVisibilityWorkflow({
    id: 'time-window-filter-visibility',
    predicates: [{kind: 'time-range', mask: outputMask}],
    output: output.ids,
    count: total,
    outputMask
  }).getCommandNodes(graph),
  // publish: clamps the count into the caller's status.
  createPublishNode<void>(graph, {
    id: 'time-window-filter-publish',
    operation: 'GPUTimeWindowFilter',
    requiredCount: total,
    output
  })
];
// No edge and no order anywhere: the graph reads them off the bindings.
graph.add(nodes);
