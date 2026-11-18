// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUFlowAggregation,
  validateFlowAggregationBindings
} from '../../src/gpu-network/flow-aggregation/gpu-flow-aggregation';
import {
  GPUNetworkFlow,
  type GPUNetworkFlowProps
} from '../../src/gpu-network/network-optimization/gpu-network-flow';
import {
  GPUNetworkFlowOperation,
  type NetworkFlowOperationProps
} from '../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {CPUNetworkFlowOperation} from '../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {constructNetworkOperationContributor} from '../utils/network-operation-test-utils';
import {
  createTerrainBranchingFlowNodes,
  GPUTerrainBranchingFlow,
  type GPUTerrainBranchingFlowProps
} from '../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

it('keeps composition helpers out of both published domain entries', async () => {
  const terrain = await import('../../src/gpu-terrain/index');
  const network = await import('../../src/gpu-network/index');
  expect(terrain).not.toHaveProperty('createTerrainBranchingFlowNodes');
  expect(network).not.toHaveProperty('validateFlowAggregationBindings');
});

it.each([
  ['internal-export', 'gpu-network/flow-aggregation/gpu-flow-aggregation.ts'],
  ['internal-export', 'gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts'],
  ['single-invocation-data-loop', 'gpu-network/network-optimization/gpu-network-flow.ts']
])('keeps %s clear in the structure-owned consumer %s', (rule, relativePath) => {
  const result = spawnSync(
    process.execPath,
    [
      resolve('scripts/check-contributor-contract.mjs'),
      '--rule',
      rule,
      '--paths',
      `modules/analysis/src/${relativePath}`
    ],
    {encoding: 'utf8', timeout: 30_000}
  );
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
});

it('internal flow preflight still rejects scratch overflow before allocating graph resources', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 1024});
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUFlowAggregation({
      zones: {kind: 'ids', zoneCount: 2},
      originZoneIds: createTransientView(graph, 'origin', 'uint32', 1),
      destinationZoneIds: createTransientView(graph, 'destination', 'uint32', 1),
      pairCapacity: 512,
      maximumProbeCount: 1,
      output: {
        ids: createTransientView(graph, 'pairs', 'uint32', 1),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => validateFlowAggregationBindings(graph, contributor)).toThrow(
      /flow-aggregation.*pair scratch.*binding/
    );
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it.each([
  'transport',
  'expected-distance'
] as const)('internal branching factory preserves %s kernels and returns the four-word private gate status', mode => {
  const device = createNullWebGPUDevice();
  try {
    const makeProps = (graph: GPUCommandGraph): GPUTerrainBranchingFlowProps => ({
      id: 'branching-parity',
      grid: {width: 3, height: 1, affine: [2, 0, 0, 0, 2, 0], pixelInterpretation: 'area'},
      fractions: createTransientView(graph, 'fractions', 'float32', 24),
      values: createTransientView(graph, 'input', 'float32', 3),
      mode,
      maximumIterations: 1,
      output: {
        values: createTransientView(graph, 'output', 'float32', 3),
        status: createTransientView(graph, 'status', 'uint32', 3),
        unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
      }
    });
    const publicGraph = new GPUCommandGraph(device);
    const contributor = new GPUTerrainBranchingFlow(makeProps(publicGraph));
    const publicNodes = contributor.getCommandNodes(publicGraph);
    const internalGraph = new GPUCommandGraph(device);
    const internalProps = makeProps(internalGraph);
    const result = createTerrainBranchingFlowNodes(internalGraph, internalProps);
    expect(getGPUCommandNodeKernelSources(result.nodes)).toEqual(
      getGPUCommandNodeKernelSources(publicNodes)
    );
    expect(result.nodes).toHaveLength(contributor.estimatedCommandNodeCount);
    expect(result.iterationStatus.format).toBe('uint32');
    expect(result.iterationStatus.length).toBe(4);
    expect(() =>
      createTerrainBranchingFlowNodes(internalGraph, {
        ...internalProps,
        maximumWalkLength: 1025
      })
    ).toThrow(/branching-parity maximumWalkLength.*1024/);
  } finally {
    device.destroy();
  }
});

it('enforces each serial reference-flow axis before accepting its documented bounds', () => {
  const device = createNullWebGPUDevice();
  try {
    const makeProps = (nodeCount: number, edgeCount: number, maximumAugmentations: number) => {
      const graph = new GPUCommandGraph(device);
      return {
        id: 'reference-bounds',
        solver: 'reference' as const,
        nodeCount,
        edges: createTransientView(graph, 'edges', 'uint32x2', edgeCount),
        capacities: createTransientView(graph, 'capacities', 'uint32', edgeCount),
        parameters: createTransientView(graph, 'parameters', 'uint32', 3),
        maximumAugmentations,
        output: {
          flow: createTransientView(graph, 'flow', 'uint32', edgeCount),
          cut: createTransientView(graph, 'cut', 'uint32', nodeCount),
          summary: createTransientView(graph, 'summary', 'uint32', 8)
        }
      };
    };
    expect(() => new GPUNetworkFlow(makeProps(129, 1, 1))).toThrow(/nodeCount.*128/);
    expect(() => new GPUNetworkFlow(makeProps(2, 1025, 1))).toThrow(/edges rows.*1024/);
    expect(() => new GPUNetworkFlow(makeProps(2, 1, 65537))).toThrow(/maximumAugmentations.*65536/);
    expect(new GPUNetworkFlow(makeProps(128, 1024, 65536)).solver).toBe('reference');
  } finally {
    device.destroy();
  }
});

it('retains validated reference constants when the caller mutates counts after construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props: GPUNetworkFlowProps = {
      nodeCount: 2,
      maximumAugmentations: 1,
      solver: 'reference',
      edges: createTransientView(graph, 'edges', 'uint32x2', 1),
      capacities: createTransientView(graph, 'capacities', 'uint32', 1),
      parameters: createTransientView(graph, 'parameters', 'uint32', 3),
      output: {
        flow: createTransientView(graph, 'flow', 'uint32', 1),
        cut: createTransientView(graph, 'cut', 'uint32', 2),
        summary: createTransientView(graph, 'summary', 'uint32', 8)
      }
    };
    const contributor = new GPUNetworkFlow(props);
    props.nodeCount = 129;
    props.maximumAugmentations = 65537;
    const [kernel] = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernel.source).toContain('const NODE_COUNT: u32 = 2u;');
    expect(kernel.source).toContain('const MAXIMUM_AUGMENTATIONS: u32 = 1u;');
  } finally {
    device.destroy();
  }
});

it('captures nested reference-flow ports without cloning their graph views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const topology = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 1)
    };
    const status = {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)};
    const output = {
      flow: createTransientView(graph, 'flow', 'uint32', 1),
      cut: createTransientView(graph, 'cut', 'uint32', 2),
      summary: createTransientView(graph, 'summary', 'uint32', 8),
      status
    };
    const props: GPUNetworkFlowProps = {
      nodeCount: 2,
      maximumAugmentations: 1,
      solver: 'reference',
      topology,
      edges: createTransientView(graph, 'edges', 'uint32x2', 1),
      capacities: createTransientView(graph, 'capacities', 'uint32', 1),
      parameters: createTransientView(graph, 'parameters', 'uint32', 3),
      output
    };
    const originalOffsets = topology.offsets;
    const originalFlow = output.flow;
    const originalInvalid = status.invalidCount;
    const originalCapacities = props.capacities;
    const contributor = new GPUNetworkFlow(props);
    topology.offsets = createTransientView(graph, 'replacement-offsets', 'uint32', 130);
    output.flow = createTransientView(graph, 'replacement-flow', 'uint32', 1);
    status.invalidCount = createTransientView(graph, 'replacement-invalid', 'uint32', 1);
    props.costs = createTransientView(graph, 'late-costs', 'uint32', 1);
    expect(contributor.props.topology!.offsets).toBe(originalOffsets);
    expect(contributor.props.output.flow).toBe(originalFlow);
    expect(contributor.props.output.status!.invalidCount).toBe(originalInvalid);
    expect(contributor.props.capacities).toBe(originalCapacities);
    expect(contributor.props.costs).toBeUndefined();
    expect(Object.isFrozen(contributor.props)).toBe(true);
    expect(Object.isFrozen(contributor.props.topology)).toBe(true);
    expect(Object.isFrozen(contributor.props.output)).toBe(true);
    expect(Object.isFrozen(contributor.props.output.status)).toBe(true);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'GPU',
  'CPU'
])('captures operation counts before deferred %s flow execution', async form => {
  const props: NetworkFlowOperationProps = {
    solver: 'reference',
    nodeCount: 2,
    edgeCount: 1,
    maximumAugmentations: 1,
    edges: 'edges',
    capacities: 'capacities',
    parameters: 'parameters',
    output: {flow: 'flow', cut: 'cut', summary: 'summary', status: {invalidCount: 'invalid'}}
  };
  const gpuOperation = new GPUNetworkFlowOperation(props);
  const cpuOperation = new CPUNetworkFlowOperation(props);
  props.nodeCount = 129;
  props.edgeCount = 1025;
  props.maximumAugmentations = 65537;
  props.costs = 'late-costs';
  props.output.flow = 'late-flow';
  props.output.status!.invalidCount = 'late-invalid';
  if (form === 'GPU') {
    const contributor = constructNetworkOperationContributor(gpuOperation, {
      edges: {format: 'uint32x2', length: 1},
      capacities: {format: 'uint32', length: 1},
      parameters: {format: 'uint32', length: 3}
    }) as GPUNetworkFlow;
    expect(contributor.props.nodeCount).toBe(2);
    expect(contributor.props.maximumAugmentations).toBe(1);
    return;
  }
  const vector = (values: Uint32Array, format: 'uint32' | 'uint32x2' = 'uint32'): GPUCPUVector => ({
    format,
    length: values.length / (format === 'uint32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'uint32x2' ? 2 : 1)}],
    pack: () => values
  });
  const rows = await cpuOperation.execute({
    inputs: {
      edges: vector(Uint32Array.of(0, 1), 'uint32x2'),
      capacities: vector(Uint32Array.of(3)),
      parameters: vector(Uint32Array.of(0, 1, 0xffffffff))
    }
  });
  expect(rows['flow']).toEqual(Uint32Array.of(3));
  expect(rows['cut']).toEqual(Uint32Array.of(1, 0));
  expect(rows['invalid']).toEqual(Uint32Array.of(0));
});

it('has no stale isochrones storage-preflight allowance in the owned manifest', () => {
  const result = spawnSync(
    process.execPath,
    [
      resolve('scripts/check-contributor-contract.mjs'),
      '--check-stale',
      '--paths',
      'modules/analysis/src/gpu-network/network-isochrones/gpu-network-isochrones.ts'
    ],
    {encoding: 'utf8', timeout: 30_000}
  );
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
});
