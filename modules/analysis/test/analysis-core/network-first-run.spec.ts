// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph';
import {GPUNetworkReachability} from '@luma-contributors/analysis/gpu-network';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {reachabilityInputs} from './network-first-run.test-utils';

it('the network quick start leaves isolated node zero unreached and writes its cost bands', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'network-first-run'});
  const offsets = createInputBuffer(device, reachabilityInputs.topology.offsets);
  const neighbors = createInputBuffer(device, reachabilityInputs.topology.neighbors);
  const weights = createInputBuffer(device, reachabilityInputs.topology.weights);
  const startNodes = createInputBuffer(device, reachabilityInputs.startNodes);
  const costLimit = createInputBuffer(device, reachabilityInputs.costLimit);
  const costBreaks = createInputBuffer(device, reachabilityInputs.costBreaks);
  const costs = createOutputBuffer(device, 3);
  const bands = createOutputBuffer(device, 3);
  const bandCounts = createOutputBuffer(device, 4);
  const converged = createOutputBuffer(device, 1);
  const iterationLimitReached = createOutputBuffer(device, 1);
  const buffers = [
    offsets,
    neighbors,
    weights,
    startNodes,
    costLimit,
    costBreaks,
    costs,
    bands,
    bandCounts,
    converged,
    iterationLimitReached
  ];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  try {
    graph.add(
      new GPUNetworkReachability({
        topology: {
          offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 4),
          neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', 1),
          weights: importGraphBuffer(graph, 'weights', weights, 'float32', 1)
        },
        startNodes: importGraphBuffer(graph, 'startNodes', startNodes, 'uint32', 1),
        costLimit: importGraphBuffer(graph, 'costLimit', costLimit, 'float32', 1),
        costBreaks: importGraphBuffer(graph, 'costBreaks', costBreaks, 'float32', 4),
        maximumIterations: 4,
        output: {
          costs: importGraphBuffer(graph, 'costs', costs, 'float32', 3),
          bands: importGraphBuffer(graph, 'bands', bands, 'uint32', 3),
          bandCounts: importGraphBuffer(graph, 'bandCounts', bandCounts, 'uint32', 4),
          status: {
            converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
            iterationLimitReached: importGraphBuffer(
              graph,
              'limit',
              iterationLimitReached,
              'uint32',
              1
            )
          }
        }
      })
    );
    compiled = await graph.compileAsync();
    submitGraph(device, compiled);
    expect(await readFloat32(costs)).toEqual([Infinity, 0, 2]);
    expect(await readUint32(bands)).toEqual([0xffffffff, 0, 1]);
    expect(await readUint32(bandCounts)).toEqual([1, 1, 0, 0]);
    expect(await readUint32(converged)).toEqual([1]);
    expect(await readUint32(iterationLimitReached)).toEqual([0]);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
