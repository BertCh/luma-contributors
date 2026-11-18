// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUKShortestPaths} from '../../../src/gpu-network/k-shortest-paths/index';
import {
  getParsedKernelSources,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {setupKShortestPaths} from './k-shortest-paths-test-utils';

it('keeps shader and pipeline counts constant as Yen rounds and queries grow', () => {
  const counts = [];
  for (const routeCount of [2, 8, 16]) {
    const {graph, contributor} = setupKShortestPaths({routeCount, queryCount: routeCount});
    const nodes = contributor.getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    counts.push({
      sources: new Set(sources.map(kernel => kernel.source)).size,
      pipelines: getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size
    });
    for (const variant of ['starts', 'edges', 'proposals', 'merge']) {
      const kernels = sources.filter(kernel => new RegExp(`-\\d+-${variant}$`).test(kernel.id));
      expect(kernels).toHaveLength(routeCount);
      expect(new Set(kernels.map(kernel => kernel.source)).size, variant).toBe(1);
      expect(kernels[0].source).toContain('let round = state[base]');
    }
  }
  expect(counts).toEqual([counts[0], counts[0], counts[0]]);
});

it('checks the aligned query bindings before declaring any scratch', () => {
  const {graph, props} = setupKShortestPaths({
    nodeCount: 2,
    queryCount: 1,
    routeCount: 2,
    maximumPathLength: 2,
    candidateCapacity: 1
  });
  const backing = createTransientView(graph, 'query-backing', 'uint32', 64).buffer;
  const sources = graph.createDataView(backing, {format: 'uint32', byteOffset: 252, length: 1});
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxStorageBufferBindingSize: 200}
  });
  const contributor = new GPUKShortestPaths({
    ...props,
    candidateCapacity: 1,
    maximumPathLength: 2,
    sources
  });
  const create = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /k-shortest-paths sources.*256.*limit 200/
  );
  expect(create).not.toHaveBeenCalled();
});

it('a reused starts shader reads the next spur round from GPU state', () => {
  const {graph, contributor} = setupKShortestPaths({
    nodeCount: 3,
    queryCount: 1,
    routeCount: 2,
    maximumPathLength: 3,
    candidateCapacity: 1
  });
  const kernel = getGPUCommandNodeKernelSources(
    contributor.getCommandNodes(graph).filter(node => node.id === 'k-shortest-paths-0-starts')
  )[0];
  // Five eight-word records after five status words. Accepted route 0 is the unit-cost chain
  // 0 -> 1 -> 2, so the next round must search both spurs with root costs zero and one.
  const state = new Uint32Array(45);
  state[0] = 1;
  state[1] = 1;
  state.set([3, 0, 1, 2, 0, 1], 6);
  const starts = new Uint32Array(2);
  const startCosts = new Float32Array(2);
  const interpreter = new WgslExec(
    new WgslParser().parse(kernel.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
  );
  interpreter.dispatchWorkgroups(
    'main',
    [1, 1, 1],
    {0: {0: state, 1: Uint32Array.of(0), 2: Float32Array.of(1, 1), 3: starts, 4: startCosts}},
    {constants: kernel.constants}
  );
  expect(Array.from(starts)).toEqual([0, 4]);
  expect(Array.from(startCosts)).toEqual([0, 1]);
});

it('builds GPU Yen above 256 vertices without graph-sized private arrays', () => {
  for (const options of [
    {nodeCount: 10000, maximumPathLength: 8},
    {nodeCount: 10000, maximumPathLength: 8, routeCount: 1},
    {nodeCount: 300, maximumPathLength: 300, routeCount: 2},
    {nodeCount: 2, maximumPathLength: 1024, routeCount: 32, candidateCapacity: 4096}
  ]) {
    const {graph, contributor} = setupKShortestPaths(options);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const sources = getParsedKernelSources(nodes);
    for (const {id: name, source: source, constants} of sources.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source, constants), name).toBeLessThanOrEqual(8192);
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    }
  }
});

it('validates named limits, command budgets, output shape, aliasing and storage before nodes', () => {
  for (const [name, value] of [
    ['routeCount', 33],
    ['maximumPathLength', 1025],
    ['candidateCapacity', 4097],
    ['maximumIterations', 1025],
    ['candidateCapacity', 0],
    ['maximumPathLength', 1.5]
  ] as const) {
    expect(() => setupKShortestPaths({[name]: value})).toThrow(
      new RegExp(`k-shortest-paths ${name}`)
    );
  }
  expect(() => setupKShortestPaths({routeCount: 32, maximumIterations: 1024})).toThrow(
    /k-shortest-paths estimated command node count.*32768/
  );
  const {props} = setupKShortestPaths();
  expect(
    () => new GPUKShortestPaths({...props, output: {...props.output, edges: props.output.vertices}})
  ).toThrow(/k-shortest-paths output views/);
  expect(() => new GPUKShortestPaths({...props, targets: props.output.counts})).toThrow(
    /k-shortest-paths output/
  );
  expect(
    () => new GPUKShortestPaths({...props, output: {...props.output, counts: props.output.lengths}})
  ).toThrow(/k-shortest-paths output.counts/);
  const oversized = setupKShortestPaths({nodeCount: 1000000, maximumPathLength: 64});
  expect(() => oversized.contributor.getCommandNodes(oversized.graph)).toThrow(
    /k-shortest-paths frontier.*limit/
  );
});

it('an overlong one-slot reconstruction leaves the neighboring query untouched', () => {
  const {graph, contributor} = setupKShortestPaths({
    nodeCount: 2,
    queryCount: 2,
    routeCount: 2,
    maximumPathLength: 1,
    candidateCapacity: 1
  });
  const nodes = contributor.getCommandNodes(graph);
  const kernel = getGPUCommandNodeKernelSources(
    nodes.filter(node => node.id === 'k-shortest-paths-0-proposals')
  )[0];
  // Two query records of 21 words each. Query zero is active; query one has an invalid source.
  // Independent shortest-path inputs select CSR edge 1, the cheaper parallel edge into node 1.
  const state = new Uint32Array(42);
  state[1] = 1;
  const interpreter = new WgslExec(
    new WgslParser().parse(kernel.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
  );
  interpreter.dispatchWorkgroups(
    'main',
    [1, 1, 1],
    {
      0: {
        0: state,
        1: Uint32Array.of(1, 0),
        2: Uint32Array.of(0, 0xffffffff),
        3: Float32Array.of(0, 1, Infinity, Infinity),
        4: Uint32Array.of(0xffffffff, 0, 0xffffffff, 0xffffffff),
        5: Uint32Array.of(0, 2, 2),
        6: Uint32Array.of(1, 1),
        7: Float32Array.of(2, 1, -1, -1)
      }
    },
    {constants: kernel.constants}
  );
  expect(state[18]).toBe(0xffffffff); // Proposal length reports the path budget stop.
  expect(Array.from(state.slice(21))).toEqual(new Array(21).fill(0));
});

it('gates every composed search pass for the 100000-vertex inactive/early-finished batch', () => {
  const {graph, contributor} = setupKShortestPaths({
    nodeCount: 100000,
    queryCount: 1,
    routeCount: 32,
    maximumPathLength: 8,
    maximumIterations: 2
  });
  const nodes = contributor.getCommandNodes(graph);
  const searches = nodes.filter(node => node.workload?.operation === 'GPUNetworkReachability');
  expect(searches.length).toBeGreaterThan(160);
  expect(searches.filter(node => node.condition?.source !== 'gpu').map(node => node.id)).toEqual(
    []
  );
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
});
