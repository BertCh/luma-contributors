// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUAdaptiveBandwidthKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {GPUSpaceTimeRelativeRisk} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-relative-risk';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

const CONTRIBUTORS = ['point', 'kernel', 'adaptive', 'space-time', 'risk'] as const;
type Contributor = (typeof CONTRIBUTORS)[number];

function createFixture(kind: Contributor, nestedScratch = false, rowCount = 8) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeExtentGrid(
    nestedScratch && (kind === 'point' || kind === 'kernel') ? 4 : 1,
    nestedScratch && (kind === 'point' || kind === 'kernel') ? 4 : 1,
    [0, 0, 1, 1]
  );
  const positions = createTransientView(graph, 'positions', 'float32x2', rowCount);
  const times = createTransientView(graph, 'times', 'float32', rowCount);
  const weights = createTransientView(graph, 'weights', 'float32', rowCount);
  const parameters = createTransientView(graph, 'parameters', 'float32', 14);
  const values = createTransientView(graph, 'values', 'float32', grid.width * grid.height);
  const controlRows = nestedScratch && kind === 'risk' ? 12 : rowCount;
  const controlPositions = createTransientView(graph, 'controls', 'float32x2', controlRows);
  const controlTimes = createTransientView(graph, 'control-times', 'float32', controlRows);
  const props = {
    positions,
    weights,
    parameters,
    grid,
    indexGridSize: (nestedScratch && kind === 'adaptive' ? [8, 8] : [2, 2]) as readonly [
      number,
      number
    ],
    indexBounds: [0, 0, 1, 1] as const,
    output: {values}
  };
  function create(overrides: {parameters?: GraphDataView<'float32'>; output?: object} = {}) {
    const common = {...props, ...overrides};
    switch (kind) {
      case 'point':
        return new GPUPointDensity({
          positions,
          weights,
          bounds: [0, 0, 1, 1],
          gridSize: [grid.width, grid.height],
          statistic: 'sum',
          output: common.output as typeof props.output
        });
      case 'kernel':
        return new GPUKernelDensity({
          ...common,
          method: nestedScratch ? 'binned' : 'exact',
          binnedPadding: nestedScratch ? 8 : 0,
          output: common.output as typeof props.output
        });
      case 'adaptive':
        return new GPUAdaptiveBandwidthKernelDensity({
          ...common,
          output: common.output as typeof props.output
        });
      case 'space-time':
        return new GPUSpaceTimeKernelDensity({
          ...common,
          times,
          sliceCount: 1,
          normalization: 'probability',
          output: common.output as typeof props.output
        });
      case 'risk':
        return new GPUSpaceTimeRelativeRisk({
          ...common,
          casePositions: positions,
          caseTimes: times,
          caseWeights: weights,
          controlPositions,
          controlTimes,
          sliceCount: 1,
          output: common.output as typeof props.output
        });
    }
  }
  return {device, graph, props, create};
}

function expectAtomicFailure(
  graph: GPUCommandGraph,
  callback: () => unknown,
  message: RegExp
): void {
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const importBuffer = vi.spyOn(graph, 'importBuffer');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(callback).toThrow(message);
    expect(allocate).not.toHaveBeenCalled();
    expect(importBuffer).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    allocate.mockRestore();
    importBuffer.mockRestore();
    add.mockRestore();
  }
}

for (const kind of CONTRIBUTORS) {
  it(`${kind} preflights caller STORAGE usage without changing the graph`, () => {
    const {device, graph, props, create} = createFixture(kind);
    try {
      const buffer = graph.createTransientBuffer({
        id: 'non-storage',
        byteLength: 4,
        usage: Buffer.COPY_DST
      });
      const values = graph.createDataView(buffer, {
        format: 'float32',
        length: props.output.values.length
      });
      const density = create({output: {values}});
      expectAtomicFailure(graph, () => density.getCommandNodes(graph), /Buffer.STORAGE/);
    } finally {
      device.destroy();
    }
  });

  it(`${kind} preflights the aligned caller binding prefix without changing the graph`, () => {
    const {device, graph, create} = createFixture(kind);
    try {
      // Four logical bytes fit the 128-byte limit; their 252-byte aligned prefix does not.
      const buffer = graph.createTransientBuffer({
        id: 'prefixed',
        byteLength: 512,
        usage: Buffer.STORAGE
      });
      const values = graph.createDataView(buffer, {format: 'float32', byteOffset: 252, length: 1});
      const density = create({output: {values}});
      device.limits.maxStorageBufferBindingSize = 128;
      expectAtomicFailure(
        graph,
        () => density.getCommandNodes(graph),
        /needs 256 bytes in one binding \(limit 128\)/
      );
    } finally {
      device.destroy();
    }
  });

  it(`${kind} preflights derived and nested scratch maxBufferSize without changing the graph`, () => {
    const {device, graph, create} = createFixture(kind, true);
    try {
      const density = create();
      device.limits.maxBufferSize = kind === 'space-time' ? 96 : 128;
      const message =
        kind === 'point'
          ? /fixed-point grid scratch byteLength/
          : kind === 'kernel'
            ? /binned fixed-point grid scratch byteLength/
            : kind === 'adaptive'
              ? /index offset and scan scratch byteLength/
              : /cell-ordered point table scratch byteLength/;
      expectAtomicFailure(graph, () => density.getCommandNodes(graph), message);
    } finally {
      device.destroy();
    }
  });
}

it('small binned KDE preflights its 48-byte blur plan independently of the grid scratch', () => {
  const {device, graph, props} = createFixture('kernel');
  try {
    const density = new GPUKernelDensity({
      ...props,
      positions: createTransientView(graph, 'one-point', 'float32x2', 1),
      weights: undefined,
      parameters: createTransientView(graph, 'short-parameters', 'float32', 9),
      method: 'binned',
      binnedPadding: 0
    });
    device.limits.maxStorageBufferBindingSize = 40;
    expectAtomicFailure(graph, () => density.getCommandNodes(graph), /binned blur plan scratch/);
  } finally {
    device.destroy();
  }
});

for (const kind of ['adaptive', 'space-time', 'risk'] as const) {
  it(`${kind} stages packed output fields sharing one imported physical buffer`, () => {
    const {device, graph, create} = createFixture(kind);
    const allocation = device.createBuffer({byteLength: 128, usage: Buffer.STORAGE});
    try {
      const handle = graph.importBuffer(
        {id: 'packed-output', byteLength: 128, usage: Buffer.STORAGE},
        allocation
      );
      const float = (byteOffset: number, length = 1) =>
        graph.createDataView(handle, {format: 'float32', byteOffset, length});
      const integer = (byteOffset: number) =>
        graph.createDataView(handle, {format: 'uint32', byteOffset, length: 1});
      const output =
        kind === 'adaptive'
          ? {
              values: float(0),
              weightSum: float(4),
              pointBandwidths: float(8, 8),
              pilotDensities: float(40, 8)
            }
          : kind === 'space-time'
            ? {values: float(0), weightSum: float(4), status: {invalidCount: integer(8)}}
            : {
                values: float(0),
                caseDensity: float(4),
                controlDensity: float(8),
                status: {caseInvalidCount: integer(12), controlInvalidCount: integer(16)}
              };
      const density = create({output});
      const nodes = density.getCommandNodes(graph);
      const publishes = nodes.filter(node => node.id.includes('-publish-'));
      expect(publishes).toHaveLength(kind === 'adaptive' ? 4 : kind === 'risk' ? 5 : 3);
      expect(getGPUCommandNodePipelineKeys(publishes, {includeShaderLayout: true}).size).toBe(1);
      expectNoMixedStorageUsage(nodes);
      for (const node of nodes.filter(node => !node.id.includes('-publish-'))) {
        for (const resource of node.resources ?? []) {
          if ('buffer' in resource && resource.buffer instanceof GraphDataView) {
            expect(resource.buffer.buffer, node.id).not.toBe(handle);
          }
        }
      }
      for (const {source, constants, id} of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(source, constants), id).toEqual([]);
      }
      // Reusing any logical bytes is still invalid, even though publishing uses separate dispatches.
      expect(() => create({output: {...output, values: float(4)}})).toThrow(/overlap/);
    } finally {
      allocation.destroy();
      device.destroy();
    }
  });
}

function expectNoMixedStorageUsage(nodes: readonly GPUCommandNode[]): void {
  for (const node of nodes) {
    const usages = new Map<unknown, Set<string>>();
    for (const resource of node.resources ?? []) {
      if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) continue;
      const handle =
        resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
      const set = usages.get(handle) ?? new Set<string>();
      set.add(resource.usage);
      usages.set(handle, set);
    }
    for (const set of usages.values()) {
      expect(set.has('storage-read') && set.has('storage-read-write'), node.id).toBe(false);
    }
  }
}

it('relative risk validates control weights and diagnostics at construction', () => {
  const {device, graph, props} = createFixture('risk');
  try {
    const times = createTransientView(graph, 'times-again', 'float32', 8);
    const base = {
      ...props,
      casePositions: props.positions,
      caseTimes: times,
      controlPositions: props.positions,
      controlTimes: times,
      sliceCount: 1
    };
    expect(
      () =>
        new GPUSpaceTimeRelativeRisk({
          ...base,
          controlWeights: createTransientView(graph, 'short-control-weights', 'float32', 7)
        })
    ).toThrow(/control.*weights length/);
    expect(
      () =>
        new GPUSpaceTimeRelativeRisk({
          ...base,
          output: {
            ...base.output,
            status: {
              controlInvalidCount: createTransientView(graph, 'wrong-status', 'float32', 1) as never
            }
          }
        })
    ).toThrow(/control.*invalidCount/);
  } finally {
    device.destroy();
  }
});

for (const kind of ['adaptive', 'space-time'] as const) {
  it(`${kind} rejects finite binary64 bounds that cannot become WGSL float32 literals at construction`, () => {
    const {device, graph, props} = createFixture(kind);
    try {
      const common = {...props, indexBounds: [0, 0, 1e39, 1] as const};
      expect(() =>
        kind === 'adaptive'
          ? new GPUAdaptiveBandwidthKernelDensity(common)
          : new GPUSpaceTimeKernelDensity({
              ...common,
              times: createTransientView(graph, 'invalid-bounds-times', 'float32', 8),
              sliceCount: 1
            })
      ).toThrow(/indexBounds\[2\].*finite/);
    } finally {
      device.destroy();
    }
  });
}

it('packed space-time density accepts a small buffer limit without unused dispatch metadata', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const density = new GPUSpaceTimeKernelDensity({
      positions: createTransientView(graph, 'one-position', 'float32x2', 1),
      times: createTransientView(graph, 'one-time', 'float32', 1),
      parameters: createTransientView(graph, 'parameters', 'float32', 14),
      grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
      sliceCount: 1,
      indexGridSize: [1, 1],
      indexBounds: [0, 0, 1, 1],
      output: {values: createTransientView(graph, 'one-value', 'float32', 1)}
    });
    device.limits.maxBufferSize = 56;
    device.limits.maxUniformBufferBindingSize = 0;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(density.getCommandNodes(graph).length).toBeGreaterThan(0);
    for (const [allocation] of allocate.mock.calls)
      expect(allocation.byteLength, allocation.id).toBeLessThanOrEqual(56);
    allocate.mockRestore();
  } finally {
    device.destroy();
  }
});

for (const kind of ['adaptive', 'space-time', 'risk'] as const) {
  it(`${kind} preflights shared-output publication uniforms before any working allocation`, () => {
    const {device, graph, create} = createFixture(kind);
    try {
      const buffer = graph.createTransientBuffer({
        id: 'shared-small-uniform',
        byteLength: 128,
        usage: Buffer.STORAGE
      });
      const float = (byteOffset: number) =>
        graph.createDataView(buffer, {format: 'float32', length: 1, byteOffset});
      const output =
        kind === 'risk'
          ? {values: float(0), caseDensity: float(4)}
          : {values: float(0), weightSum: float(4)};
      const density = create({output});
      device.limits.maxUniformBufferBindingSize = 63;
      expectAtomicFailure(
        graph,
        () => density.getCommandNodes(graph),
        /maxUniformBufferBindingSize/
      );
    } finally {
      device.destroy();
    }
  });
}

for (const kind of ['adaptive', 'space-time', 'risk'] as const) {
  it(`${kind} bounds every nested radix, scan and reduction allocation by the preflight layout`, () => {
    const rowCount = 257; // First radix-sized input, with a partial final workgroup.
    const {device, graph, create} = createFixture(kind, false, rowCount);
    try {
      const density = create();
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const nodes = density.getCommandNodes(graph);
      const largestScratchByteLength = (kind === 'adaptive' ? 4 : 16) * rowCount;
      expect(allocate.mock.calls.length).toBeGreaterThan(10);
      for (const [allocation] of allocate.mock.calls) {
        expect(allocation.byteLength, allocation.id).toBeLessThanOrEqual(largestScratchByteLength);
      }
      for (const node of nodes) {
        for (const resource of node.resources ?? []) {
          if ('buffer' in resource && resource.usage.startsWith('storage-')) {
            const buffer =
              resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
            expect(buffer.usage & Buffer.STORAGE, node.id).not.toBe(0);
          }
        }
      }
      allocate.mockRestore();
    } finally {
      device.destroy();
    }
  });
}
