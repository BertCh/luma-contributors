// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationUint32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_RASTER_D8_DIRECTIONS} from '../../raster-analysis/cost-distance/raster-grid-utils';
import {GPU_TERRAIN_FLOW_STREAM_UNRESOLVED} from './gpu-terrain-flow';
import type {GPUTerrainFlowDirectionEncoding} from './terrain-flow-tracing';
import {
  getTerrainDrainageDirection,
  getTerrainDrainageReceiver,
  TERRAIN_DRAINAGE_NONE,
  TERRAIN_DRAINAGE_UNRESOLVED,
  validateTerrainDrainageOptions
} from './cpu-terrain-drainage-utils';
import {
  GPUTerrainStreamOrderOperation,
  type TerrainStreamOrderOperationProps
} from './gpu-terrain-stream-order-operation';

export type CPUTerrainStreamOrderResult = {
  streamOrder: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
  iterationCount: Uint32Array;
};

/**
 * Computes Strahler orders with a donor queue. Unknown direction neighbors seed unknown order,
 * which propagates downstream. Invalid/non-stream cells are zero. Rootless stream components
 * (including tributaries into receiver cycles) are unresolved, matching the GPU Euler-tour path.
 * The streams input uses 0 for nonstream, 1..0xfffffffe for membership and 0xffffffff for
 * unknown. Translate imported unresolved 0xfffffffe to 0xffffffff, as on the GPU port; this
 * preserves the existing meaning of 0xfffffffe membership. Direction sentinels are separate.
 *
 * What differs from the GPU contributor:
 * - Completes the donor queue without maximumIterations, the 1024-round pull cap, or tour scratch.
 * - iterationCount is 0 (no GPU rounds); iterationLimitReached is always 0. converged is 0 only
 *   for rootless receiver-cycle components, matching the Euler-tour path's status definition.
 * - Rootless tributaries match the Euler-tour output; the GPU pull fallback can resolve their
 *   upstream orders before it stalls at a cycle. Unknown-input statuses follow the tour path too.
 * - Resolved orders are exact integers; there is no numeric tolerance difference.
 */
export function computeTerrainStreamOrderOnCPU(
  flowDirections: Uint32Array,
  streams: Uint32Array,
  options: {grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding}
): CPUTerrainStreamOrderResult {
  const {grid, directionEncoding} = options;
  const count = validateTerrainDrainageOptions(
    'computeTerrainStreamOrderOnCPU',
    grid,
    directionEncoding
  );
  if (flowDirections.length !== count)
    throw new Error('flowDirections must contain one word per grid cell');
  if (streams.length !== count) throw new Error('streams must contain one word per grid cell');
  const isStream = (cell: number) =>
    streams[cell] !== 0 && flowDirections[cell] !== TERRAIN_DRAINAGE_NONE;
  const receivers = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const pending = new Uint32Array(count);
  const maximumOrders = new Uint32Array(count);
  const maximumCounts = new Uint32Array(count);
  const unknown = new Uint8Array(count);
  const streamOrder = new Uint32Array(count);
  for (let cell = 0; cell < count; cell++) {
    if (!isStream(cell)) continue;
    streamOrder[cell] = TERRAIN_DRAINAGE_NONE;
    const receiver = getTerrainDrainageReceiver(
      cell,
      getTerrainDrainageDirection(flowDirections[cell], directionEncoding),
      grid.width,
      grid.height
    );
    if (receiver !== TERRAIN_DRAINAGE_NONE && isStream(receiver)) {
      receivers[cell] = receiver;
      pending[receiver]++;
    }
    unknown[cell] = Number(
      streams[cell] === GPU_TERRAIN_FLOW_STREAM_UNRESOLVED ||
        flowDirections[cell] === TERRAIN_DRAINAGE_UNRESOLVED
    );
    for (const direction of GPU_RASTER_D8_DIRECTIONS) {
      const neighbor = getTerrainDrainageReceiver(cell, direction.code, grid.width, grid.height);
      if (
        neighbor !== TERRAIN_DRAINAGE_NONE &&
        isStream(neighbor) &&
        flowDirections[neighbor] === TERRAIN_DRAINAGE_UNRESOLVED
      )
        unknown[cell] = 1;
    }
  }
  const queue: number[] = [];
  for (let cell = 0; cell < count; cell++)
    if (isStream(cell) && pending[cell] === 0) queue.push(cell);
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    const order =
      maximumOrders[cell] === 0 ? 1 : maximumOrders[cell] + Number(maximumCounts[cell] >= 2);
    streamOrder[cell] = unknown[cell] ? TERRAIN_DRAINAGE_NONE : order;
    const receiver = receivers[cell];
    if (receiver === TERRAIN_DRAINAGE_NONE) continue;
    unknown[receiver] |= unknown[cell];
    if (order > maximumOrders[receiver]) {
      maximumOrders[receiver] = order;
      maximumCounts[receiver] = 1;
    } else if (order === maximumOrders[receiver]) maximumCounts[receiver]++;
    if (--pending[receiver] === 0) queue.push(receiver);
  }
  // A tour has no position for any cell whose downstream path never reaches a root. Classify
  // those paths separately so the CPU agrees with the tour even for tributaries into a cycle.
  const state = new Uint8Array(count);
  const rootless = new Uint8Array(count);
  let hasCycle = false;
  for (let start = 0; start < count; start++) {
    if (!isStream(start) || state[start] === 2) continue;
    const path: number[] = [];
    let cell = start;
    while (cell !== TERRAIN_DRAINAGE_NONE && state[cell] === 0) {
      state[cell] = 1;
      path.push(cell);
      cell = receivers[cell];
    }
    const reachesCycle =
      cell !== TERRAIN_DRAINAGE_NONE && (state[cell] === 1 || rootless[cell] !== 0);
    if (reachesCycle) hasCycle = true;
    for (const member of path) {
      state[member] = 2;
      rootless[member] = Number(reachesCycle);
      if (reachesCycle) streamOrder[member] = TERRAIN_DRAINAGE_NONE;
    }
  }
  return {
    streamOrder,
    converged: new Uint32Array([Number(!hasCycle)]),
    iterationLimitReached: new Uint32Array(1),
    iterationCount: new Uint32Array(1)
  };
}

/** Exact CPU stream-order twin, with the GPU tour's unresolved-cell semantics. */
export class CPUTerrainStreamOrderOperation extends GPUCPUOperation {
  constructor(props: TerrainStreamOrderOperationProps) {
    const operation = new GPUTerrainStreamOrderOperation(props);
    const {id} = operation;
    validateTerrainDrainageOptions(id, props.grid, props.directionEncoding);
    if (props.maximumIterations !== undefined)
      validateCapacity(
        id,
        'maximumIterations',
        props.maximumIterations,
        1,
        Number.MAX_SAFE_INTEGER
      );
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const result = computeTerrainStreamOrderOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'flowDirections'),
          getCPUOperationUint32Rows(id, inputs, 'streams'),
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTerrainStreamOrderResult]
          ])
        );
      }
    });
  }
}
