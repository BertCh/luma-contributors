// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUSpatialInteraction,
  type GPUSpatialInteractionProps
} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {GPUSpatialInteractionOperation} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createProps(
  graph: GPUCommandGraph,
  pairs = 8192,
  groups = 257
): GPUSpatialInteractionProps {
  return {
    origins: createTransientView(graph, 'origins', 'uint32', pairs),
    destinations: createTransientView(graph, 'destinations', 'uint32', pairs),
    flows: createTransientView(graph, 'flows', 'float32', pairs),
    covariate: createTransientView(graph, 'covariate', 'float32', pairs),
    pairState: createTransientView(graph, 'pair-state', 'uint32', pairs),
    originCount: groups,
    destinationCount: groups,
    coefficients: [-1, 0, 1],
    constraint: 'doubly',
    maximumIterations: 3,
    output: {
      predictions: createTransientView(graph, 'predictions', 'float32', pairs),
      summary: createTransientView(graph, 'summary', 'float32', 4),
      status: createTransientView(graph, 'status', 'uint32', 1),
      candidates: createTransientView(graph, 'candidates', 'float32', 12)
    }
  };
}

it('selects sparse beyond either reference cap, retains explicit dense caps, and forwards method', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const [pairs, groups] of [
      [8192, 2],
      [4, 257]
    ]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, pairs, groups);
      const contributor = new GPUSpatialInteraction(props);
      expect(contributor.method).toBe('sparse');
      expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
        contributor.estimatedCommandNodeCount
      );
      expect(() => new GPUSpatialInteraction({...props, method: 'dense'})).toThrow(/4096|128/);
    }
    const contributor = constructNetworkOperationContributor(
      new GPUSpatialInteractionOperation({
        method: 'sparse',
        origins: 'origins',
        destinations: 'destinations',
        flows: 'flows',
        covariate: 'covariate',
        pairCount: 4,
        originCount: 2,
        destinationCount: 2,
        coefficients: [0],
        constraint: 'doubly',
        maximumIterations: 2,
        output: {
          predictions: 'predictions',
          summary: 'summary',
          candidates: 'candidates',
          status: 'status'
        }
      }),
      {
        origins: {format: 'uint32', length: 4},
        destinations: {format: 'uint32', length: 4},
        flows: {format: 'float32', length: 4},
        covariate: {format: 'float32', length: 4}
      }
    ) as GPUSpatialInteraction;
    expect(contributor.method).toBe('sparse');
  } finally {
    device.destroy();
  }
});

it('keeps pipeline count constant as round and candidate counts grow; audits generated WGSL', () => {
  const device = createNullWebGPUDevice();
  try {
    const counts: number[] = [];
    for (const [iterations, candidates] of [
      [1, 1],
      [7, 3]
    ]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, 513, 129);
      props.maximumIterations = iterations;
      props.coefficients = [-1, 0, 1].slice(0, candidates);
      props.output.candidates = createTransientView(
        graph,
        'candidate-output',
        'float32',
        candidates * 4
      );
      const contributor = new GPUSpatialInteraction({...props, method: 'sparse'});
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      counts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      const sums = getGPUCommandNodeKernelSources(
        nodes.filter(node => node.workload?.variant === 'sparse-compensated-sum')
      );
      expect(sums.length).toBeGreaterThan(0);
      expect(
        sums.every(kernel => kernel.source.includes('compensation += getAdditionRoundingError'))
      ).toBe(true);
      const repeated = getGPUCommandNodeKernelSources(
        nodes.filter(node => node.workload?.variant === 'sparse-scale')
      );
      expect(new Set(repeated.map(kernel => kernel.source)).size).toBe(1);
    }
    expect(counts[0]).toBeGreaterThan(0);
    expect(counts[0]).toBe(counts[1]);
  } finally {
    device.destroy();
  }
});

it('rejects aligned writable aliasing, nonfinite float32 literals and excessive graph budgets', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 4, 2);
    const output = createTransientView(graph, 'combined', 'float32', 64);
    expect(
      () =>
        new GPUSpatialInteraction({
          ...props,
          output: {
            ...props.output,
            predictions: graph.createDataView(output.buffer, {format: 'float32', length: 4}),
            summary: graph.createDataView(output.buffer, {
              format: 'float32',
              length: 4,
              byteOffset: 16
            })
          }
        })
    ).toThrow(/aligned binding/);
    expect(() => new GPUSpatialInteraction({...props, coefficients: [Number.MAX_VALUE]})).toThrow(
      /coefficients.*finite/
    );
    expect(() => new GPUSpatialInteraction({...props, tolerance: Number.MIN_VALUE})).toThrow(
      /tolerance.*finite/
    );
    expect(
      () => new GPUSpatialInteraction({...props, method: 'sparse', maximumIterations: 4096})
    ).toThrow(/estimated command node count/);
  } finally {
    device.destroy();
  }
});

it('rejects oversized aligned input and scratch bindings before declaring any resources', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 4, 2);
    const large = createTransientView(graph, 'large', 'float32', 20);
    props.flows = graph.createDataView(large.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 64
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const contributor = new GPUSpatialInteraction({...props, method: 'sparse'});
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/flows.*80.*64/);
    expect(createBuffer).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('checks the sparse dispatch gate before allocation even when small marginal sorts need fewer buckets', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUSpatialInteraction({...createProps(graph, 4, 2), method: 'sparse'});
    Object.assign(device.limits, {maxStorageBufferBindingSize: 48});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/balancing dispatch.*60.*48/);
    expect(allocate).not.toHaveBeenCalled();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('accepts packed origin and destination residuals that are written in separate dispatches', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const method of ['dense', 'sparse'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, 4, 2);
      const storage = createTransientView(graph, 'packed-output-storage', 'uint32', 320);
      const residualStorage = createTransientView(graph, 'packed-residual-storage', 'float32', 4);
      const floating = (byteOffset: number, length: number) =>
        graph.createDataView(storage.buffer, {format: 'float32', byteOffset, length});
      const contributor = new GPUSpatialInteraction({
        ...props,
        method,
        output: {
          predictions: floating(0, 4),
          summary: floating(256, 4),
          status: graph.createDataView(storage.buffer, {
            format: 'uint32',
            byteOffset: 512,
            length: 1
          }),
          candidates: method === 'dense' ? props.output.candidates : floating(768, 12),
          originResiduals: graph.createDataView(residualStorage.buffer, {
            format: 'float32',
            length: 2
          }),
          destinationResiduals: graph.createDataView(residualStorage.buffer, {
            format: 'float32',
            length: 2,
            byteOffset: 8
          })
        }
      });
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('rejects co-bound read-only and writable outputs of one physical buffer at disjoint aligned offsets', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const method of ['dense', 'sparse'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, 4, 2);
      const storage = createTransientView(graph, 'residual-usage-scope', 'uint32', 128);
      const status = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
      const originResiduals = graph.createDataView(storage.buffer, {
        format: 'float32',
        length: 2,
        byteOffset: 256
      });
      expect(
        () =>
          new GPUSpatialInteraction({
            ...props,
            method,
            output: {...props.output, status, originResiduals}
          })
      ).toThrow(/physical buffers/);
      if (method === 'dense') {
        const candidates = graph.createDataView(storage.buffer, {
          format: 'float32',
          length: 12,
          byteOffset: 256
        });
        expect(
          () =>
            new GPUSpatialInteraction({
              ...props,
              method,
              output: {...props.output, status, candidates}
            })
        ).toThrow(/physical buffers/);
      }
    }
  } finally {
    device.destroy();
  }
});
