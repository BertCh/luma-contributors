// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {
  GPUMapMatching,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('S1 replays matching graphs against the original retained device table allocations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const offsets = createInputBuffer(device, Uint32Array.of(0, 1, 2, 3, 3));
  const neighbors = createInputBuffer(device, Uint32Array.of(1, 2, 3));
  const weights = createInputBuffer(device, Float32Array.of(1, 1, 1));
  const nodes = createInputBuffer(device, Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0));
  const cutoff = createInputBuffer(device, Float32Array.of(0));
  const points = createInputBuffer(device, Float32Array.of(0.5, 0, 2.5, 0));
  const tracks = createInputBuffer(device, Uint32Array.of(0, 2));
  const parameters = createInputBuffer(
    device,
    getGPUMapMatchingParameterValues({
      sigma: 0.1,
      beta: 1,
      searchRadius: 0.1,
      routeFactor: 3,
      routeSlack: 0
    })
  );
  const tableOffsets = createOutputBuffer(device, 5),
    tableTargets = createOutputBuffer(device, 16),
    tableCosts = createOutputBuffer(device, 16),
    built = createOutputBuffer(device, 1),
    complete = createOutputBuffer(device, 1),
    matched = createOutputBuffer(device, 2),
    breaks = createOutputBuffer(device, 2);
  const buffers = [
    offsets,
    neighbors,
    weights,
    nodes,
    cutoff,
    points,
    tracks,
    parameters,
    tableOffsets,
    tableTargets,
    tableCosts,
    built,
    complete,
    matched,
    breaks
  ];
  const build = () => {
    const graph = new GPUCommandGraph(device);
    graph.add(
      new GPUNetworkCostTable({
        topology: {
          offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 5),
          neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', 3),
          weights: importGraphBuffer(graph, 'weights', weights, 'float32', 3)
        },
        costLimit: importGraphBuffer(graph, 'cutoff', cutoff, 'float32', 1),
        output: {
          offsets: importGraphBuffer(graph, 'table-offsets', tableOffsets, 'uint32', 5),
          targets: importGraphBuffer(graph, 'table-targets', tableTargets, 'uint32', 16),
          costs: importGraphBuffer(graph, 'table-costs', tableCosts, 'float32', 16),
          builtCostLimit: importGraphBuffer(graph, 'built', built, 'float32', 1),
          status: {complete: importGraphBuffer(graph, 'complete', complete, 'uint32', 1)}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
    } finally {
      compiled.destroy();
    }
  };
  const match = async () => {
    const graph = new GPUCommandGraph(device);
    graph.add(
      new GPUMapMatching({
        topology: {
          offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 5),
          neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', 3),
          nodePositions: importGraphBuffer(graph, 'nodes', nodes, 'float32x2', 4)
        },
        positions: importGraphBuffer(graph, 'points', points, 'float32x2', 2),
        trackOffsets: importGraphBuffer(graph, 'tracks', tracks, 'uint32', 2),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 8),
        candidateCount: 1,
        cellSize: 1,
        bounds: {minimum: [-1, -1], maximum: [4, 1]},
        routeTable: {
          offsets: importGraphBuffer(graph, 'table-offsets', tableOffsets, 'uint32', 5),
          targets: importGraphBuffer(graph, 'table-targets', tableTargets, 'uint32', 16),
          costs: importGraphBuffer(graph, 'table-costs', tableCosts, 'float32', 16),
          costLimit: importGraphBuffer(graph, 'built', built, 'float32', 1),
          status: {complete: importGraphBuffer(graph, 'complete', complete, 'uint32', 1)}
        },
        output: {
          matchedEdges: importGraphBuffer(graph, 'matched', matched, 'uint32', 2),
          breaks: importGraphBuffer(graph, 'breaks', breaks, 'uint32', 2)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      return Array.from(await readUint32(breaks, 2));
    } finally {
      compiled.destroy();
    }
  };
  try {
    build();
    expect(Array.from(await readUint32(complete, 1))).toEqual([1]);
    cutoff.write(Float32Array.of(10));
    // Both matches import the original physical table buffers; no table readback/re-upload.
    expect(await match()).toEqual([0, 0]);
    expect(await match()).toEqual([0, 0]);
    expect(Array.from(await readFloat32(built, 1))).toEqual([0]);
    build();
    expect(Array.from(await readUint32(complete, 1))).toEqual([1]);
    expect(Array.from(await readFloat32(built, 1))).toEqual([10]);
    expect(await match()).toEqual([0, 0]);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});
