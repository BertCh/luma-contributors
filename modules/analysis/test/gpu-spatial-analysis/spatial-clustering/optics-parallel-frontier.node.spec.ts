// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUOPTICS,
  GPU_OPTICS_MAXIMUM_COMMAND_NODE_COUNT,
  type GPUOPTICSProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {computeOPTICSOnCPU} from '../../../src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import fixture from './optics-parallel-fixtures.json';

let resourceSequence = 0;

function createOPTICS(
  graph: GPUCommandGraph,
  rows: number,
  overrides: Partial<GPUOPTICSProps> = {}
) {
  const id = overrides.id ?? 'parallel-check';
  const resourcePrefix = `${id}-${resourceSequence++}`;
  return new GPUOPTICS({
    id,
    positions: createTransientView(graph, `${resourcePrefix}-positions`, 'float32x2', rows),
    minimumSamples: 2,
    maximumDistance: 10,
    extractionMethod: 'xi',
    minimumClusterSize: 2,
    neighborIndex: 'grid',
    orderingMethod: 'parallel',
    output: {
      ordering: createTransientView(graph, `${resourcePrefix}-ordering`, 'uint32', rows),
      coreDistances: createTransientView(graph, `${resourcePrefix}-core`, 'float32', rows),
      reachability: createTransientView(graph, `${resourcePrefix}-reachability`, 'float32', rows),
      predecessors: createTransientView(graph, `${resourcePrefix}-predecessors`, 'uint32', rows),
      labels: createTransientView(graph, `${resourcePrefix}-labels`, 'uint32', rows),
      status: createTransientView(graph, `${resourcePrefix}-status`, 'uint32', 3),
      ports: {
        status: {
          candidateOverflow: createTransientView(graph, `${resourcePrefix}-overflow`, 'uint32', 1),
          iterationCount: createTransientView(graph, `${resourcePrefix}-iterations`, 'uint32', 1),
          converged: createTransientView(graph, `${resourcePrefix}-converged`, 'uint32', 1),
          iterationLimitReached: createTransientView(graph, `${resourcePrefix}-limit`, 'uint32', 1)
        }
      }
    },
    ...overrides
  });
}

it('accepts xi beyond 4096 through the indexed cap, and validates the parallel path at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(() => createOPTICS(graph, 4097)).not.toThrow();
    expect(createOPTICS(graph, 65536).estimatedCommandNodeCount).toBe(
      GPU_OPTICS_MAXIMUM_COMMAND_NODE_COUNT
    );
    expect(() => createOPTICS(graph, 65537)).toThrow(/^parallel-check positions rows.*65536/);
    expect(() => createOPTICS(graph, 256, {neighborIndex: 'brute-force'})).toThrow(
      /orderingMethod 'parallel' needs neighborIndex 'grid'/
    );
    expect(() => createOPTICS(graph, 256, {orderingMethod: 'unknown' as never})).toThrow(
      /orderingMethod must be/
    );
    expect(() =>
      createOPTICS(graph, 65536, {orderingMethod: 'sequential', frontierBatchSize: 1})
    ).toThrow(/estimated command node count.*32768/);
    expect(() => createOPTICS(graph, 256, {maximumIterations: 257})).toThrow(
      /maximumIterations.*256/
    );
    expect(() => createOPTICS(graph, 256, {maximumDistance: Infinity})).toThrow(
      /^parallel-check maximumDistance/
    );
    expect(() => createOPTICS(graph, 256, {neighborCapacity: 0})).toThrow(
      /^parallel-check neighborCapacity/
    );
  } finally {
    device.destroy();
  }
});

it('bounds actual node counts at reduction boundaries and reuses row scratch between visits', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const rows of [2, 255, 256, 257, 4097, 65536]) {
      const graph = new GPUCommandGraph(device);
      // Full 4097-row graph crosses the shared 32768-node budget; cap the largest graph
      // at two visits to check its storage shapes without allocating 590,000 test nodes.
      const optics = createOPTICS(graph, rows, {maximumIterations: rows === 65536 ? 2 : rows});
      const nodes = optics.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(optics.estimatedCommandNodeCount);
      const frontier = nodes.filter(
        node => node.id.includes('-parallel-') && !node.id.endsWith('-initialize')
      );
      const visits = optics.props.maximumIterations ?? rows;
      expect(frontier.length).toBe(visits * (rows > 256 ? 9 : 7));
      for (const node of frontier) {
        if (node.id.includes('-parallel-commit-')) expect(node.condition, node.id).toBeUndefined();
        else expect(node.condition?.source, node.id).toBe('gpu');
      }
      const xiBatches = nodes.filter(node => node.workload?.variant === 'xi-extraction');
      const xiGates = nodes.filter(node => node.workload?.variant === 'xi-gate');
      expect(xiGates.length).toBe(xiBatches.length);
      expect(xiBatches.every(node => node.condition?.source === 'gpu')).toBe(true);
      expect(xiGates.every(node => node.condition === undefined)).toBe(true);
      const keyBuffers = new Set(
        frontier.flatMap(node =>
          (node.resources ?? []).flatMap(resource => {
            if (!('buffer' in resource)) return [];
            const buffer = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
            return buffer.id.includes('-parallel-keys') ? [buffer.id] : [];
          })
        )
      );
      expect(keyBuffers.size).toBe(rows > 256 ? 3 : 2);
      if (rows === 257) {
        // Compile one visit's templates and xi batches on the null device, then parse all
        // WGSL independently. This catches storage binding and shader syntax regressions.
        const templates = nodes.filter(
          node =>
            !node.id.includes('-parallel-') ||
            node.id.endsWith('-0') ||
            node.id.endsWith('-initialize')
        );
        const sources = getGPUCommandNodeKernelSources(templates);
        for (const kernel of sources) {
          const reflection = new WgslReflect(kernel.source);
          expect(reflection.storage.length, kernel.id).toBeLessThanOrEqual(8);
        }
      }
    }
  } finally {
    device.destroy();
  }
});

it('never storage-writes a dispatch buffer in the dispatch it gates', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const orderingMethod of ['sequential', 'parallel'] as const) {
      const graph = new GPUCommandGraph(device);
      const nodes = createOPTICS(graph, 257, {
        orderingMethod,
        maximumIterations: 2
      }).getCommandNodes(graph);
      const conflictingNodes: string[] = [];
      for (const node of nodes) {
        if (node.condition?.source !== 'gpu') continue;
        const indirectBuffer = node.condition.buffer;
        for (const resource of node.resources ?? []) {
          if (
            !('buffer' in resource) ||
            !['storage-write', 'storage-read-write'].includes(resource.usage)
          )
            continue;
          const storageBuffer =
            'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
          if (storageBuffer === indirectBuffer) conflictingNodes.push(node.id);
        }
      }
      // WebGPU usage scope forbids writable storage + indirect on the same buffer,
      // including when storage and the indirect triple are disjoint ranges.
      expect(conflictingNodes, orderingMethod).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('the CPU twin matches the independent scikit-learn fixture above 4096, with both predecessor modes', () => {
  for (const extraction of fixture.xi) {
    const result = computeOPTICSOnCPU(Float32Array.from(fixture.positions), {
      ...fixture.options,
      extractionMethod: 'xi',
      predecessorCorrection: extraction.predecessorCorrection,
      orderingMethod: 'parallel',
      neighborIndex: 'grid'
    });
    expect(Array.from(result.ordering)).toEqual(fixture.ordering);
    expect(Array.from(result.predecessors)).toEqual(fixture.predecessors);
    expect(Array.from(result.coreDistances)).toEqual(fixture.coreDistances);
    expect(Array.from(result.reachability)).toEqual(
      fixture.reachability.map(value => (value === 'Infinity' ? Infinity : value))
    );
    expect(Array.from(result.labels)).toEqual(extraction.labels);
    expect(Array.from(result.clusterSummary)).toEqual([extraction.intervals.length, 0]);
    expect(
      Array.from(result.clusterIntervals.subarray(0, extraction.intervals.length * 2))
    ).toEqual(extraction.intervals.flat());
    expect(
      result.clusterIntervals
        .subarray(extraction.intervals.length * 2)
        .every(value => value === 0xffffffff)
    ).toBe(true);
    expect(Array.from(result.iterationCount)).toEqual([fixture.ordering.length]);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  }
});
