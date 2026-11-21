// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  // Cell 0 drains to invalid cell 1. Cell 2 is a definite terminal with no reach to start.
  {
    name: 'malformed direction',
    direction: 3,
    stream: 1,
    unresolved: [1, 1, 0],
    reachStarts: [0, 0, 0]
  },
  {
    name: 'out-of-grid direction',
    direction: 64,
    stream: 1,
    unresolved: [1, 1, 0],
    reachStarts: [0, 0, 0]
  },
  {
    name: 'invalid direction',
    direction: 0xffffffff,
    stream: 1,
    unresolved: [1, 1, 0],
    reachStarts: [0, 0, 0]
  },
  // With an unknown direction, cell 1 may donate to either neighbor.
  {
    name: 'unresolved direction',
    direction: 0xfffffffe,
    stream: 1,
    unresolved: [1, 1, 1],
    reachStarts: [0, 0, 0]
  },
  // With unknown membership, cell 1 and its upstream cell 0 are unresolved. Cell 2's
  // donor count is also uncertain because cell 1 points east to it.
  {
    name: 'unknown mask 0xffffffff',
    direction: 1,
    stream: 0xffffffff,
    unresolved: [1, 1, 1],
    reachStarts: [0, 0, 0]
  },
  {
    name: 'unknown mask 0xfffffffe',
    direction: 1,
    stream: 0xfffffffe,
    unresolved: [1, 1, 1],
    reachStarts: [0, 0, 0]
  },
  // Only headwater 0 starts a reach in the definite 0 -> 1 -> 2 chain.
  {name: 'definite chain', direction: 1, stream: 1, unresolved: [0, 0, 0], reachStarts: [1, 0, 0]}
])('stream network classifies a three-cell row with $name', ({
  direction,
  stream,
  unresolved,
  reachStarts
}) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const id = 'three-cell-stream-network';
    const words = (name: string, length: number) =>
      createTransientView(graph, name, 'uint32', length);
    const directions = words('directions', 3);
    const streams = words('streams', 3);
    const nodes = new GPUTerrainStreamNetwork({
      id,
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      flowDirections: directions,
      streams,
      output: {
        sourceCells: words('source', 3),
        targetCells: words('target', 3),
        vertexOffsets: words('offsets', 4),
        positions: createTransientView(graph, 'positions', 'float32x2', 6),
        lengths: createTransientView(graph, 'lengths', 'float32', 3),
        reachCount: words('reach-count', 1),
        vertexCount: words('vertex-count', 1),
        status: {overflow: words('overflow', 1)},
        unresolvedCount: words('unresolved-count', 1)
      }
    }).getCommandNodes(graph);
    const classifyIndex = nodes.findIndex(node => node.id === `${id}-reaches`);
    const classificationViews = (nodes[classifyIndex]?.resources ?? []).flatMap(resource =>
      'buffer' in resource && resource.buffer instanceof GraphDataView ? [resource.buffer] : []
    );
    const reachFlags = classificationViews.find(view => view.buffer.id === `${id}-reach-flags`);
    const unresolvedFlags = classificationViews.find(
      view => view.buffer.id === `${id}-unresolved-flags`
    );
    if (!reachFlags || !unresolvedFlags) {
      throw new Error('Expected stream-network classification buffers');
    }

    const memory = createCPUGraphMemory();
    memory.write(directions, Uint32Array.of(1, direction, 0));
    memory.write(streams, Uint32Array.of(1, stream, 1));
    // The last kernel marks one flag per reach start and one per unresolved stream cell.
    executeCommandNodesOnCPU(nodes.slice(0, classifyIndex + 1), memory);
    expect(Array.from(memory.readUint32(reachFlags))).toEqual(reachStarts);
    expect(Array.from(memory.readUint32(unresolvedFlags))).toEqual(unresolved);
  } finally {
    device.destroy();
  }
});
