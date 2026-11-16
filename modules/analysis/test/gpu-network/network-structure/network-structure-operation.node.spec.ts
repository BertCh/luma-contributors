// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkStructure} from '../../../src/gpu-network/network-structure/gpu-network-structure';
import {GPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {computeNetworkStructureOnCPU} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createComponentCSR} from './strong-components-fixtures';

it('operation lowering forwards mode, SCC budget and each status port into executed WGSL', () => {
  const input = createComponentCSR(7, [
    [0, 1],
    [1, 2],
    [2, 0],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 5]
  ]);
  for (const [mode, strongComponentsMaximumIterations] of [
    ['parallel', 1],
    ['parallel', 32],
    ['exact', 1]
  ] as const) {
    const operation = new GPUNetworkStructureOperation({
      mode,
      strongComponentsMaximumIterations,
      nodeCount: 7,
      edgeCount: input.neighbors.length,
      topology: {offsets: 'offsets-value', neighbors: 'neighbors-value'},
      output: {strongComponents: 'labels-value'},
      strongComponentsStatus: {
        converged: 'converged-value',
        iterationCount: 'iterations-value',
        iterationLimitReached: 'exhausted-value'
      }
    });
    expect(operation.outputs['converged']).toEqual({
      value: 'converged-value',
      format: 'uint32',
      length: 1
    });
    expect(operation.outputs['iterationCount']).toEqual({
      value: 'iterations-value',
      format: 'uint32',
      length: 1
    });
    expect(operation.outputs['iterationLimitReached']).toEqual({
      value: 'exhausted-value',
      format: 'uint32',
      length: 1
    });
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const views: Record<string, GraphDataView> = {
      offsets: createTransientView(graph, 'offsets', 'uint32', input.offsets.length),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', input.neighbors.length)
    };
    for (const [port, output] of Object.entries(operation.outputs))
      views[port] = createTransientView(graph, port, 'uint32', output.length);
    const contributor = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(port: string) => views[port] as GraphDataView<Format>
    );
    expect(contributor).toBeInstanceOf(GPUNetworkStructure);
    if (!(contributor instanceof GPUNetworkStructure))
      throw new Error('Expected the network structure contributor');
    expect(contributor.mode).toBe(mode);
    expect(contributor.strongComponentsMaximumIterations).toBe(strongComponentsMaximumIterations);
    const memory = createCPUGraphMemory();
    memory.write(views['offsets'], input.offsets);
    memory.write(views['neighbors'], input.neighbors);
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    const finished = mode === 'exact' || strongComponentsMaximumIterations > 1;
    expect(Array.from(memory.readUint32(views['strongComponents']))).toEqual(
      finished
        ? Array.from(computeNetworkStructureOnCPU(input).strongComponents)
        : new Array(7).fill(0xffffffff)
    );
    expect(Array.from(memory.readUint32(views['converged']))).toEqual([Number(finished)]);
    expect(Array.from(memory.readUint32(views['iterationLimitReached']))).toEqual([
      Number(!finished)
    ]);
    const iterationCount = memory.readUint32(views['iterationCount'])[0];
    expect(iterationCount).toBeGreaterThanOrEqual(1);
    expect(iterationCount).toBeLessThanOrEqual(strongComponentsMaximumIterations);
    operation.destroy();
  }
});

it('operation status ports reject graph value aliases at construction', () => {
  expect(
    () =>
      new GPUNetworkStructureOperation({
        nodeCount: 1,
        edgeCount: 0,
        topology: {offsets: 'offsets', neighbors: 'neighbors'},
        output: {strongComponents: 'labels'},
        strongComponentsStatus: {
          converged: 'labels',
          iterationCount: 'iterations',
          iterationLimitReached: 'exhausted'
        }
      })
  ).toThrow(/network-structure outputs.converged.*no other port uses/);
});
