// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPURankMarkov,
  type GPURankMarkovProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {GPUTransitionDiagnostics} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {GPUTransitionMatrix} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix';
import {GPURankMarkovOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createProps(graph: GPUCommandGraph, rows = 129, periods = 3): GPURankMarkovProps {
  return {
    values: createTransientView(graph, 'values', 'float32', rows * periods),
    periodCount: periods,
    output: {
      ranks: createTransientView(graph, 'ranks', 'uint32', rows * periods),
      counts: createTransientView(graph, 'counts', 'uint32', rows * rows),
      mobility: createTransientView(graph, 'mobility', 'float32', periods - 1)
    }
  };
}

it('sorts above 64 states and keeps the named pairwise reference cap and diagnostic cap', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const contributor = new GPURankMarkov(props);
    expect(contributor.rankingMethod).toBe('sorted');
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.some(node => node.workload?.variant === 'rank-scatter')).toBe(true);
    expect(nodes.some(node => node.workload?.variant === 'rank')).toBe(false);
    expect(() => new GPURankMarkov({...props, rankingMethod: 'pairwise'})).toThrow(
      /rank-markov.*rowCount.*64/
    );
    expect(
      () =>
        new GPURankMarkov({
          ...props,
          output: {
            ...props.output,
            steadyStates: createTransientView(graph, 'steady', 'float32', 129)
          }
        })
    ).toThrow(/classCount.*64/);
    expect(() =>
      constructNetworkOperationContributor(
        new GPURankMarkovOperation({
          values: 'values',
          rowCount: 129,
          periodCount: 3,
          rankingMethod: 'pairwise',
          output: {ranks: 'ranks', counts: 'counts'}
        }),
        {values: {format: 'float32', length: 387}}
      )
    ).toThrow(/rowCount.*64/);
  } finally {
    device.destroy();
  }
});

it('audits every generated kernel and keeps custom shader count constant as periods grow', () => {
  const device = createNullWebGPUDevice();
  try {
    const counts: number[] = [];
    for (const periods of [2, 8, 64]) {
      const graph = new GPUCommandGraph(device);
      const nodes = new GPURankMarkov({
        ...createProps(graph, 129, periods),
        mode: 'geographic-rank'
      }).getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of kernels)
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      counts.push(
        new Set(
          kernels
            .filter(kernel => kernel.operation === 'GPURankMarkov')
            .map(
              kernel =>
                `${kernel.source}${JSON.stringify(kernel.constants)}${JSON.stringify(kernel.shaderLayout)}`
            )
        ).size
      );
    }
    expect(counts[0]).toBeGreaterThan(0);
    expect(new Set(counts).size).toBe(1);
  } finally {
    device.destroy();
  }
});

it('rejects writable aliasing across ranks, counts, mobility and nested status', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 4, 2);
    const combined = createTransientView(graph, 'combined', 'uint32', 64);
    const ranks = graph.createDataView(combined.buffer, {format: 'uint32', length: 8});
    // Logical ranges do not overlap; their aligned bindings do.
    const counts = graph.createDataView(combined.buffer, {
      format: 'uint32',
      length: 16,
      byteOffset: 32
    });
    expect(() => new GPURankMarkov({...props, output: {...props.output, ranks, counts}})).toThrow(
      /physical buffers/
    );
    expect(
      () =>
        new GPURankMarkov({
          ...props,
          output: {
            ...props.output,
            status: {
              converged: graph.createDataView(props.output.ranks.buffer, {
                format: 'uint32',
                length: 1
              })
            },
            steadyStates: createTransientView(graph, 'steady', 'float32', 4)
          }
        })
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('checks actual aligned bindings before declaring any scratch', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 4, 2);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const large = createTransientView(graph, 'large', 'float32', 24);
    props.values = graph.createDataView(large.buffer, {
      format: 'float32',
      length: 8,
      byteOffset: 64
    });
    const contributor = new GPURankMarkov(props);
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/values.*96.*64/);
    expect(createBuffer).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('preflights integer mobility totals and nested reduction carries before declaring any resources', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const periods of [2, 4]) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPURankMarkov(createProps(graph, 1, periods));
      Object.assign(device.limits, {maxStorageBufferBindingSize: periods * 4});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        periods === 2 ? /mobility tile carries.*12.*8/ : /mobility totals.*24.*16/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('bounds total transitions before uint32 histograms or row totals can wrap', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(
      () =>
        new GPUTransitionMatrix({
          classes: createTransientView(graph, 'classes', 'uint32', 2 ** 33),
          periodCount: 2,
          classCount: 2,
          output: {counts: createTransientView(graph, 'counts', 'uint32', 4)}
        })
    ).toThrow(/rowCount.*periodCount.*4294967295/);
  } finally {
    device.destroy();
  }
});

it('rejects a nested convergence output aliasing an input mask', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 4, 2);
    const mask = createTransientView(graph, 'mask', 'uint32', 4);
    expect(
      () =>
        new GPURankMarkov({
          ...props,
          mask,
          output: {
            ...props.output,
            steadyStates: createTransientView(graph, 'steady', 'float32', 4),
            status: {converged: graph.createDataView(mask.buffer, {format: 'uint32', length: 1})}
          }
        })
    ).toThrow(/outputs.*inputs/);
  } finally {
    device.destroy();
  }
});

it('preflights standalone diagnostic inputs and nested chain scratch before any resource declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const counts = createTransientView(graph, 'diagnostic-counts', 'uint32', 32);
    const contributor = new GPUTransitionDiagnostics({
      counts,
      classCount: 4,
      conditionCount: 2,
      output: {steadyStates: createTransientView(graph, 'steady', 'float32', 12)}
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    Object.assign(device.limits, {maxStorageBufferBindingSize: 160});
    expect(() => contributor.getCommandNodes(graph)).toThrow(/chainWork.*192.*160/);
    expect(allocate).not.toHaveBeenCalled();
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    Object.defineProperty(counts.buffer, 'usage', {value: Buffer.COPY_DST});
    expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('ignores unwritten optional diagnostic status views in the new preflight and output checks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph, 8, 2);
    const unused = {
      converged: undefined,
      iterationLimitReached: graph.createDataView(props.output.counts.buffer, {
        format: 'uint32',
        length: 1
      })
    };
    const contributor = new GPURankMarkov({...props, output: {...props.output, status: unused}});
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('accepts packed ranks and mobility with disjoint logical ranges across separate dispatches', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph, 8, 2);
    const storage = createTransientView(graph, 'rank-and-mobility', 'uint32', 17);
    const contributor = new GPURankMarkov({
      ...props,
      output: {
        ...props.output,
        ranks: graph.createDataView(storage.buffer, {format: 'uint32', length: 16}),
        mobility: graph.createDataView(storage.buffer, {
          format: 'float32',
          byteOffset: 64,
          length: 1
        })
      }
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
