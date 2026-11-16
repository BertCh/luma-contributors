// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  getViewBindingRange
} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createFixture(maximumCandidates: number, withStatus = true) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `facility-status-${sequence++}`, format, length);
  const status = {
    invalidCount: view('uint32', 1),
    iterationCount: view('uint32', 1),
    converged: view('uint32', 1),
    iterationLimitReached: view('uint32', 1)
  };
  const props = {
    id: 'facility-status',
    operation: 'capacitated-p-median' as const,
    facilityCount: 4,
    selectCount: 2,
    maximumCandidates,
    costs: view('float32', 40),
    demand: view('float32', 10),
    capacities: view('float32', 4),
    threshold: view('float32', 1),
    output: {
      selectedFacilities: view('uint32', 4),
      assignments: view('uint32', 10),
      objective: view('float32', 1),
      summary: view('uint32', 4),
      status: withStatus ? status : undefined
    }
  };
  return {device, graph, props, status, view};
}

it('estimates every candidate batch, reduction and optional named-status command exactly', () => {
  for (const maximumCandidates of [0, 1, 63, 64, 65, 65536, 65537, 1048576]) {
    for (const withStatus of [false, true]) {
      const {device, graph, props} = createFixture(maximumCandidates, withStatus);
      try {
        const contributor = new GPUNetworkFacilityOptimization(props);
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
        expect(nodes.length).toBeLessThanOrEqual(131);
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
          expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        }
      } finally {
        device.destroy();
      }
    }
  }
});

it('validates scalar status shapes, aliases and graph ownership before creating kernels', () => {
  const {device, graph, props, view, status} = createFixture(1);
  try {
    for (const invalidCount of [view('uint32', 0), view('uint32', 2)]) {
      expect(
        () =>
          new GPUNetworkFacilityOptimization({
            ...props,
            output: {...props.output, status: {invalidCount}}
          })
      ).toThrow(/facility-status output.status.invalidCount must hold one row/);
    }
    expect(
      () =>
        new GPUNetworkFacilityOptimization({
          ...props,
          output: {...props.output, status: {invalidCount: props.capacities as never}}
        })
    ).toThrow(/facility-status/);
    expect(
      () =>
        new GPUNetworkFacilityOptimization({
          ...props,
          output: {
            ...props.output,
            status: {converged: status.invalidCount, invalidCount: status.invalidCount}
          }
        })
    ).toThrow(/facility-status.*distinct/);
    expect(
      () =>
        new GPUNetworkFacilityOptimization({
          ...props,
          output: {...props.output, status: {converged: props.output.summary}}
        })
    ).toThrow(/facility-status/);
    const otherGraph = new GPUCommandGraph(device);
    const foreign = createTransientView(otherGraph, 'foreign-status', 'uint32', 1);
    const contributor = new GPUNetworkFacilityOptimization({
      ...props,
      output: {...props.output, status: {invalidCount: foreign}}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(/facility-status.*graph/);
  } finally {
    device.destroy();
  }
});

it('checks the candidate score binding against the device before allocating search scratch', () => {
  const {device, graph, props} = createFixture(65536);
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 65536});
    const contributor = new GPUNetworkFacilityOptimization(props);
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /facility-status candidate scores needs 1048576 bytes.*limit 65536/
    );
  } finally {
    device.destroy();
  }
});

for (const operation of ['backup-coverage', 'p-dispersion', 'capacitated-p-median'] as const) {
  it(`${operation} never binds zero-byte buffers for empty demand`, () => {
    const {device, graph, props, view} = createFixture(4);
    let sequence = 0;
    const empty = <Format extends 'uint32' | 'float32'>(format: Format) =>
      graph.createDataView(
        graph.createTransientBuffer({
          id: `empty-facility-${sequence++}`,
          byteLength: 0,
          usage: Buffer.STORAGE
        }),
        {format, length: 0}
      );
    try {
      recordWGSLKernelNodes(graph);
      const contributor = new GPUNetworkFacilityOptimization({
        ...props,
        operation,
        facilityCount: 2,
        selectCount: 2,
        costs: operation === 'p-dispersion' ? view('float32', 4) : empty('float32'),
        demand: empty('float32'),
        capacities: view('float32', 2),
        output: {
          ...props.output,
          selectedFacilities: view('uint32', 2),
          assignments: empty('uint32')
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      for (const node of nodes) {
        // Published compute nodes and parameter uploads expose their views through resources;
        // they need not have a createWGSLKernelNode recording entry.
        expect(node.resources?.length).toBeGreaterThan(0);
        for (const resource of node.resources ?? []) {
          if (!('buffer' in resource) || !(resource.buffer instanceof GraphDataView)) continue;
          const binding = resource.buffer;
          const range = getViewBindingRange(binding);
          expect(range.offset + range.size, `${node.id}: ${binding.buffer.id}`).toBeLessThanOrEqual(
            binding.buffer.byteLength
          );
        }
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  });
}

it('checks actual aligned input/output bindings before declaring any facility search nodes', () => {
  for (const field of ['costs', 'assignments'] as const) {
    const {device, graph, props} = createFixture(1);
    try {
      recordWGSLKernelNodes(graph);
      const original = field === 'costs' ? props.costs : props.output.assignments;
      const storage = createTransientView(graph, `offset-${field}`, original.format, 128);
      const offsetView = graph.createDataView(storage.buffer, {
        format: original.format,
        length: original.length,
        byteOffset: 252
      });
      const contributor = new GPUNetworkFacilityOptimization({
        ...props,
        ...(field === 'costs' ? {costs: offsetView as typeof props.costs} : {}),
        output: {
          ...props.output,
          ...(field === 'assignments'
            ? {assignments: offsetView as typeof props.output.assignments}
            : {})
        }
      });
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 256});
      expect(original.length * 4).toBeLessThanOrEqual(256);
      expect(getViewBindingRange(offsetView).size).toBeGreaterThan(256);
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /facility-status input\/output aligned binding range.*limit 256/
      );
      expect(getWGSLKernelNodeRecord(graph, 'facility-status-initialize')).toBeUndefined();
    } finally {
      device.destroy();
    }
  }
});

it('preflights fixed control scratch even for one-candidate facility searches', () => {
  const {device, graph, props, view} = createFixture(1);
  try {
    recordWGSLKernelNodes(graph);
    const contributor = new GPUNetworkFacilityOptimization({
      ...props,
      facilityCount: 1,
      selectCount: 1,
      costs: view('float32', 1),
      demand: view('float32', 1),
      capacities: view('float32', 1),
      output: {
        ...props.output,
        selectedFacilities: view('uint32', 1),
        assignments: view('uint32', 1)
      }
    });
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 16});
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /facility-status facility control state needs 28 bytes.*limit 16/
    );
    expect(getWGSLKernelNodeRecord(graph, 'facility-status-initialize')).toBeUndefined();
  } finally {
    device.destroy();
  }
});

it('shared facility reductions retain one pipeline and dispatch each level at its own size', () => {
  for (const [maximumCandidates, expectedWorkgroups] of [
    [64, [1]],
    [65, [2, 1]],
    [4096, [64, 1]],
    [65536, [1024, 16, 1]]
  ] as const) {
    const {device, graph, props} = createFixture(maximumCandidates);
    try {
      const nodes = new GPUNetworkFacilityOptimization(props).getCommandNodes(graph);
      const reductions = nodes.filter(node => node.workload?.variant === 'candidate-reduction');
      expect(reductions.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        expectedWorkgroups
      );
      const sources = getGPUCommandNodeKernelSources(reductions);
      expect(
        new Set(sources.map(kernel => JSON.stringify([kernel.source, kernel.constants]))).size
      ).toBe(1);
      for (const kernel of sources)
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    } finally {
      device.destroy();
    }
  }
});
