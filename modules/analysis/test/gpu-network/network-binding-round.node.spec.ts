// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  getViewBindingRange,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {NETWORK_CASES} from '../analysis-core/kernel-audit/cases-network';
import {MOB10_FLOW_MODELS_AUDIT_CASES} from '../analysis-core/kernel-audit/cases-mob10-flow-models';
import {MOB05_MOB09_AUDIT_CASES} from '../analysis-core/kernel-audit/cases-mob05-mob09-edge-volumes-time-geography';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  type KernelAuditCase
} from '../analysis-core/kernel-audit/kernel-audit-context';
import {GPUNetworkNeighborhood} from '../../src/gpu-network/network-analysis/gpu-network-neighborhood';
import {GPUNetworkServiceAreas} from '../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {GPUMapMatchedEdgeVolumes} from '../../src/gpu-network/flow-aggregation/gpu-map-matched-edge-volumes';
import {GPURadiationFlowModel} from '../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model';

const BINDING_CASES: KernelAuditCase[] = [
  ...[
    'GPUNetworkAnalyticsColumns',
    'GPUNetworkNeighborhood',
    'GPUNetworkPathExtraction',
    'GPUNetworkServiceAreas',
    'GPUNetworkStatistics',
    'GPUFlowAggregation'
  ].map(exportName => NETWORK_CASES.find(candidate => candidate.exportName === exportName)!),
  ...MOB10_FLOW_MODELS_AUDIT_CASES,
  ...MOB05_MOB09_AUDIT_CASES.filter(
    candidate => candidate.exportName === 'GPUMapMatchedEdgeVolumes'
  )
];

/** Captures the real constructor fixture, leaving scheduling and scratch declaration for the test. */
function captureFixture(fixture: KernelAuditCase, offset: number) {
  const context = createKernelAuditContext(`binding-${fixture.exportName}`);
  const views: GraphDataView[] = [];
  const originalView = context.view;
  context.view = (format, length, name) => {
    const backing = originalView(format, length + (offset ? 1 : 0), name);
    const view = context.graph.createDataView(backing.buffer, {format, length, byteOffset: offset});
    views.push(view);
    return view;
  };
  let contributor: GPUCommandNodeProducer | undefined;
  const stop = new Error('captured contributor');
  const add = vi.spyOn(context.graph, 'add').mockImplementation(value => {
    contributor = value as GPUCommandNodeProducer;
    throw stop;
  });
  try {
    expect(() => fixture.build(context)).toThrow(stop);
  } finally {
    add.mockRestore();
  }
  expect(contributor).toBeDefined();
  return {context, views, contributor: contributor!};
}

for (const fixture of BINDING_CASES) {
  it(`${fixture.exportName}: ${fixture.name}: every caller view requires STORAGE before mutation`, () => {
    const {context, views, contributor} = captureFixture(fixture, 0);
    const create = vi.spyOn(context.graph, 'createTransientBuffer');
    try {
      for (const view of views) {
        const usage = view.buffer.usage;
        Object.defineProperty(view.buffer, 'usage', {value: Buffer.COPY_DST, configurable: true});
        expect(() => context.graph.add(contributor)).toThrow(/STORAGE/);
        expect(create).not.toHaveBeenCalled();
        expect(getDeclaredNodes(context)).toHaveLength(0);
        Object.defineProperty(view.buffer, 'usage', {value: usage, configurable: true});
      }
    } finally {
      context.device.destroy();
    }
  });

  it(`${fixture.exportName}: ${fixture.name}: checks aligned caller ranges before mutation`, () => {
    const {context, views, contributor} = captureFixture(fixture, 4);
    // All logical ranges fit; their unaligned prefix adds four bytes to the actual binding.
    const maximumLogicalBytes = Math.max(...views.map(view => view.length * view.rowByteLength));
    expect(Math.max(...views.map(view => getViewBindingRange(view).size))).toBeGreaterThan(
      maximumLogicalBytes
    );
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: maximumLogicalBytes});
    const create = vi.spyOn(context.graph, 'createTransientBuffer');
    try {
      expect(() => context.graph.add(contributor)).toThrow(/bytes in one binding \(limit/);
      expect(create).not.toHaveBeenCalled();
      expect(getDeclaredNodes(context)).toHaveLength(0);
    } finally {
      context.device.destroy();
    }
  });
}

for (const serviceAreas of [false, true]) {
  it(`${serviceAreas ? 'service areas' : 'neighborhood'} preflights composed frontier scratch before mutation`, () => {
    const context = createKernelAuditContext('frontier-scratch');
    const {graph, view} = context;
    const topology = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const contributor = serviceAreas
      ? new GPUNetworkServiceAreas({
          topology,
          facilities: view('uint32', 1),
          maximumIterations: 2,
          maximumLabelIterations: 2,
          output: {assignments: view('uint32', 2), predecessors: view('uint32', 2)}
        })
      : new GPUNetworkNeighborhood({
          topology,
          startNodes: view('uint32', 1),
          hops: view('uint32', 1),
          maximumHops: 2,
          output: {hopDistances: view('uint32', 2)}
        });
    // Caller bindings and queues fit. The phase reset binds two 256-byte dispatch slots.
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 128});
    const create = vi.spyOn(graph, 'createTransientBuffer');
    try {
      expect(() => graph.add(contributor)).toThrow(
        /(?:Dispatch|dispatch).*bytes in one binding \(limit 128/
      );
      expect(create).not.toHaveBeenCalled();
      expect(getDeclaredNodes(context)).toHaveLength(0);
    } finally {
      context.device.destroy();
    }
  });
}

it('edge volumes preflights transition table scratch before declaring conversion buffers', () => {
  const context = createKernelAuditContext('transition-scratch');
  const {graph, view} = context;
  const contributor = new GPUMapMatchedEdgeVolumes({
    traversedEdges: {edges: view('uint32', 8), counts: view('uint32', 2)},
    trackCount: 2,
    traversedEdgeCapacity: 4,
    edgeCount: 4,
    transitionPairCapacity: 64,
    output: {
      edgeVolumes: view('uint32', 4),
      transitions: {
        ids: view('uint32', 2),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    }
  });
  Object.assign(context.device.limits, {maxStorageBufferBindingSize: 128});
  const create = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => graph.add(contributor)).toThrow(/pair scratch.*bytes in one binding \(limit 128/);
    expect(create).not.toHaveBeenCalled();
    expect(getDeclaredNodes(context)).toHaveLength(0);
  } finally {
    context.device.destroy();
  }
});

it('radiation rejects shared physical input/output buffers and duplicate imports before mutation', () => {
  const context = createKernelAuditContext('physical-alias');
  const {device, graph, view} = context;
  const physical = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  const handle = graph.importBuffer(
    {id: 'physical', byteLength: 1024, usage: Buffer.STORAGE},
    physical
  );
  const positions = graph.createDataView(handle, {format: 'float32x2', length: 2});
  const duplicate = graph.importBuffer(
    {id: 'duplicate-import', byteLength: 1024, usage: Buffer.STORAGE},
    physical
  );
  const flows = graph.createDataView(duplicate, {format: 'float32', length: 4, byteOffset: 256});
  const populations = view('float32', 2);
  const outflows = view('float32', 2);
  try {
    expect(
      () => new GPURadiationFlowModel({positions, populations, outflows, output: {flows}})
    ).toThrow(/must not share buffers with inputs/);
    const duplicatePopulations = graph.createDataView(duplicate, {
      format: 'float32',
      length: 2,
      byteOffset: 512
    });
    const contributor = new GPURadiationFlowModel({
      positions,
      populations: duplicatePopulations,
      outflows,
      output: {flows: view('float32', 4)}
    });
    const create = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => graph.add(contributor)).toThrow(/physical buffer must be imported once/);
    expect(create).not.toHaveBeenCalled();
    expect(getDeclaredNodes(context)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('edge volumes accepts disjoint packed outputs written by its separate aggregation dispatches', () => {
  const context = createKernelAuditContext('packed-edge-volumes');
  const {graph, view} = context;
  const storage = view('uint32', 8).buffer;
  try {
    expect(() =>
      graph.add(
        new GPUMapMatchedEdgeVolumes({
          traversedEdges: {edges: view('uint32', 8), counts: view('uint32', 2)},
          trackCount: 2,
          traversedEdgeCapacity: 4,
          edgeCount: 4,
          transitionPairCapacity: 8,
          output: {
            edgeVolumes: graph.createDataView(storage, {format: 'uint32', length: 4}),
            transitions: {
              ids: graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 16}),
              status: {count: view('uint32', 1), overflow: view('uint32', 1)}
            }
          }
        })
      )
    ).not.toThrow();
  } finally {
    context.device.destroy();
  }
});
