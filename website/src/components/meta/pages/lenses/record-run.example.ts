// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createGPUComputationRecord,
  createGPUComputationCaptures,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  GPUSnapshotArena,
  GPUSnapshotTap
} from '@luma-contributors/analysis/computation-record';

// example:hide-start
declare const device: Device;
/** A graph that already holds an operation's kernels. */
declare const graph: GPUCommandGraph<void>;
/** The operation's output; its buffer was created with Buffer.COPY_SRC. */
declare const density: GraphDataView;
// example:hide-end
// 1. Add a copy node last. It reads the output into a buffer the CPU can map.
const arena = new GPUSnapshotArena(device, {
  id: 'run-snapshots',
  slots: [
    {name: 'density', byteLength: density.length * density.rowByteLength, elementType: 'f32'}
  ],
  capacity: 1
});
graph.add(new GPUSnapshotTap({id: 'run-snapshot-tap', arena, views: {density}}));

// 2. Compile once. The compiled graph publishes its order, buffers and hazards.
const compiled = graph.compile();

// 3. Describe it. Reads only the public id, stats and preflight report.
const description = describeGPUCommandGraph(compiled);

// 4. Assemble the record: the graph, the device and the domain.
const record = createGPUComputationRecord({
  id: 'kernel-density',
  adapter: getGPUComputationAdapter(device),
  domains: ['gpu-spatial-analysis'],
  graphs: [description]
});

// 5. Encode and submit once, then read the tap's copy.
const commandEncoder = device.createCommandEncoder();
compiled.encode(commandEncoder, {parameters: undefined});
device.submit(commandEncoder.finish());
const snapshots = await arena.read({
  graph: description.graph.id,
  resources: {density: 'density'}
});
const captures = createGPUComputationCaptures({
  executionId: record.executionId,
  snapshots,
  runId: record.runId
});
// example:hide-start
export {captures};
// example:hide-end
