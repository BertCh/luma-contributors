// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUGather, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
/** The actual two-stage adapter shared by the host declaration and headless composition tests. */
export function addStopEventGathers<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  timestamps: GraphDataView<'uint32x2'>,
  fixOnes: GraphDataView<'uint32'>,
  starts: GraphDataView<'uint32'>,
  eventTimes: GraphDataView<'uint32x2'>,
  live: GraphDataView<'uint32'>
): void {
  graph.add(
    new GPUGather({id: 'stop-event-times', source: timestamps, indices: starts, output: eventTimes})
  );
  graph.add(new GPUGather({id: 'stop-validity', source: fixOnes, indices: starts, output: live}));
}
