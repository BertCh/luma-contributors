// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, getViewBindingRange} from '@luma.gl/graph/gpu-core';
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
        expect(nodes.length).toBeLessThanOrEqual(83);
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
        const record = getWGSLKernelNodeRecord(graph, node.id);
        expect(record).toBeDefined();
        for (const {view: binding} of record!.bindings) {
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
