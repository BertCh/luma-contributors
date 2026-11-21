// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {NETWORK_CASES} from '../analysis-core/kernel-audit/cases-network';
import {
  createKernelAuditContext,
  getDeclaredNodes
} from '../analysis-core/kernel-audit/kernel-audit-context';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {GPUNetworkKernelDensity} from '../../src/gpu-network/gpu-network-kernel-density';

it('KDE rejects overlapping output fields at construction and missing STORAGE before scratch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const distances = createTransientView(graph, 'distances', 'float32', 1);
  const quadratureLengths = createTransientView(graph, 'lengths', 'float32', 1);
  const intensity = createTransientView(graph, 'intensity', 'float32', 1);
  const kernelMass = createTransientView(graph, 'mass', 'float32', 1);
  const validity = createTransientView(graph, 'validity', 'uint32', 1);
  const props = {
    distances,
    quadratureLengths,
    bandwidth: 1,
    eventCount: 1,
    output: {intensity, kernelMass, validity}
  };
  const aliased = graph.createDataView(intensity.buffer, {format: 'uint32', length: 1});
  expect(
    () => new GPUNetworkKernelDensity({...props, output: {...props.output, validity: aliased}})
  ).toThrow(/network-kernel-density.*overlap/);
  const buffer = graph.createTransientBuffer({
    id: 'copy-only',
    byteLength: 4,
    usage: Buffer.COPY_DST
  });
  const copyOnly = graph.createDataView(buffer, {format: 'float32', length: 1});
  const copyOnlyUint32 = graph.createDataView(buffer, {format: 'uint32', length: 1});
  const costsComplete = createTransientView(graph, 'costs-complete', 'uint32', 1);
  const contributor = new GPUNetworkKernelDensity({...props, distances: copyOnly});
  const create = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => graph.add(contributor)).toThrow(/network-kernel-density distances.*STORAGE/);
  const certificate = new GPUNetworkKernelDensity({
    ...props,
    costsComplete,
    output: {...props.output, status: {complete: copyOnlyUint32}}
  });
  expect(() => graph.add(certificate)).toThrow(
    /network-kernel-density output\.status\.complete.*STORAGE/
  );
  const count = new GPUNetworkKernelDensity({
    ...props,
    output: {...props.output, status: {invalidCount: copyOnlyUint32}}
  });
  expect(() => graph.add(count)).toThrow(
    /network-kernel-density output\.status\.invalidCount.*STORAGE/
  );
  const invalid = new GPUNetworkKernelDensity({
    ...props,
    output: {...props.output, validity: copyOnlyUint32}
  });
  expect(() => graph.add(invalid)).toThrow(/network-kernel-density output\.validity.*STORAGE/);
  expect(create).not.toHaveBeenCalled();
});

// Reuse declaration fixtures, but stop before scheduling the contributor. Mark caller handles
// COPY_DST-only afterwards so constructors still validate the original shapes and capacities.
for (const exportName of [
  'GPUAdjacencyMatrix',
  'GPUAdjacencyMatrixOrder',
  'GPUEdgeBundling',
  'GPUMapMatching',
  'GPUNetworkAccessibility',
  'GPUNetworkCostMatrix',
  'GPUNetworkCostTable',
  'GPUNetworkCostWeights',
  'GPUNetworkLineGraph',
  'GPUNetworkNoding',
  'GPUNetworkReachability',
  'GPUNetworkSnapping',
  'GPUNetworkKFunction',
  'GPUNetworkSubgraphFilter',
  'GPUKShortestPaths'
]) {
  it(`${exportName} rejects non-STORAGE views without declaring scratch or nodes`, () => {
    const context = createKernelAuditContext(`review-${exportName}`);
    const fixture = NETWORK_CASES.find(candidate => candidate.exportName === exportName)!;
    expect(fixture).toBeDefined();
    const views: GraphDataView[] = [];
    const originalView = context.view;
    context.view = (format, length, name) => {
      const view = originalView(format, length, name);
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
      add.mockRestore();
      expect(contributor?.getCommandNodes).toBeTypeOf('function');
      for (const view of views)
        Object.defineProperty(view.buffer, 'usage', {value: Buffer.COPY_DST});
      const create = vi.spyOn(context.graph, 'createTransientBuffer');
      expect(() => context.graph.add(contributor!)).toThrow(/STORAGE/);
      expect(create).not.toHaveBeenCalled();
      expect(getDeclaredNodes(context)).toHaveLength(0);
    } finally {
      add.mockRestore();
      context.device.destroy();
    }
  });
}

it('KDE accepts packed outputs written separately and audits scaled quotient shaders', () => {
  for (const bandwidth of [3e38, 3e-38]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const packed = createTransientView(graph, 'packed', 'uint32', 66).buffer;
    const intensity = graph.createDataView(packed, {format: 'float32', length: 2});
    const kernelMass = graph.createDataView(packed, {format: 'float32', length: 1, byteOffset: 8});
    const validity = graph.createDataView(packed, {format: 'uint32', length: 2, byteOffset: 256});
    const props = {
      distances: createTransientView(graph, 'distances', 'float32', 2),
      quadratureLengths: createTransientView(graph, 'lengths', 'float32', 2),
      eventCount: 1,
      bandwidth,
      output: {intensity, kernelMass, validity}
    };
    try {
      const nodes = new GPUNetworkKernelDensity(props).getCommandNodes(graph);
      for (const {id: name, source, constants} of getParsedKernelSources(nodes).configurations) {
        expect(auditWGSLModule(source, constants), name).toEqual([]);
        expect(source).toContain('fn getScaledQuotient(');
      }
      expect(getParsedKernelSources(nodes).get('network-kernel-density-intensity')).toContain(
        'shapeMass'
      );
      expect(() => nodes.forEach(node => graph.add(node))).not.toThrow();
      const overlapping = graph.createDataView(packed, {
        format: 'uint32',
        length: 2,
        byteOffset: 12
      });
      expect(
        () =>
          new GPUNetworkKernelDensity({...props, output: {...props.output, validity: overlapping}})
      ).toThrow(/aligned binding byte ranges must not overlap/);
      const invalidCount = graph.createDataView(packed, {
        format: 'uint32',
        length: 1,
        byteOffset: 12
      });
      expect(
        () =>
          new GPUNetworkKernelDensity({
            ...props,
            output: {...props.output, status: {invalidCount}}
          })
      ).toThrow(/output\.status\.invalidCount.*output\.validity/);
      for (const invalid of [Number.NaN, 1e100, Number.MIN_VALUE]) {
        expect(() => new GPUNetworkKernelDensity({...props, bandwidth: invalid})).toThrow(
          /network-kernel-density.*bandwidth.*float32/
        );
      }
    } finally {
      graph.device.destroy();
    }
  }
});
