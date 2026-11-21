// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  addStopValueCubeRecipe,
  addExcessRiskRecipe,
  addSpatialRateRecipe,
  addZonalDistributionRecipe,
  addPointsInPolygonsChoroplethRecipe,
  computePointsInPolygonsChoroplethOnCPU
} from '../../../src/gpu-spatial-analysis/recipes/index';

function context() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 3),
    neighbors: view('uint32', 2),
    weights: view('float32', 2)
  };
  const props = {
    time: 'start' as const,
    family: 'quadbin' as const,
    resolution: 1,
    dictionary: view('uint32x2', 2),
    weights,
    timestamps: view('uint32x2', 3),
    temporalContext: {
      unit: 'second' as const,
      epoch: 'unix' as const,
      encoding: 'int64-words' as const,
      axis: {kind: 'irregular' as const}
    },
    periods: {kind: 'fixed' as const, origin: 0n, width: 10, count: 2},
    stops: {
      centroids: view('float32x2', 2),
      startRows: view('uint32', 2),
      finishRows: view('uint32', 2),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  };
  const nodes = () => (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
  return {device, graph, view, props, nodes};
}
it('B1 audits every generated stop-cube shader for reserved identifiers', () => {
  const {device, graph, props, nodes} = context();
  try {
    addStopValueCubeRecipe(graph, props);
    const kernels = getGPUCommandNodeKernelSources(nodes());
    expect(kernels.some(kernel => kernel.id.endsWith('-stop-time'))).toBe(true);
    for (const kernel of kernels)
      expect(
        auditWGSLModule(kernel.source, kernel.constants).filter(
          finding => finding.check === 'reserved word'
        ),
        kernel.id
      ).toEqual([]);
  } finally {
    device.destroy();
  }
});
it.each([
  'timestamps',
  'startRows',
  'finishRows'
] as const)('B2 rejects nonpacked %s before declaration', field => {
  const {device, graph, props} = context();
  try {
    const original = field === 'timestamps' ? props.timestamps : props.stops[field];
    const buffer = graph.createTransientBuffer({id: 'strided', byteLength: 64, usage: 32});
    const strided = graph.createDataView(buffer, {
      format: original.format,
      length: original.length,
      byteStride: original.rowByteLength + 4
    });
    const input =
      field === 'timestamps'
        ? {...props, timestamps: strided}
        : {...props, stops: {...props.stops, [field]: strided}};
    expect(() => addStopValueCubeRecipe(graph, input as typeof props)).toThrow(/packed|stride/);
  } finally {
    device.destroy();
  }
});
it('B3 captures inherited stop and slice evidence for publication and suppression', () => {
  const {device, graph, props, view, nodes} = context();
  try {
    const sliceOverflow = view('uint32', 1);
    const result = addStopValueCubeRecipe(graph, {
      ...props,
      stops: {...props.stops, status: Object.create(props.stops.status)},
      sliceStatuses: [Object.create({candidateOverflow: sliceOverflow})]
    });
    const stopsStageStatus = result.status.stages[0].status;
    expect('count' in stopsStageStatus).toBe(true);
    expect('count' in stopsStageStatus ? stopsStageStatus.count : undefined).toBe(
      props.stops.status.count
    );
    expect(result.status.stages[0].status.overflow).toBe(props.stops.status.overflow);
    expect(result.status.stages[1].status.candidateOverflow).toBe(sliceOverflow);
    const memory = createCPUGraphMemory();
    memory.write(props.stops.status.overflow, Uint32Array.of(1));
    executeCommandNodesOnCPU(
      nodes().filter(node => node.id.startsWith('stop-value-cube-overflow-')),
      memory
    );
    expect(Array.from(memory.readUint32(result.status.overflow))).toEqual([1]);
  } finally {
    device.destroy();
  }
});
it('B4 preserves supplied weights metadata and singleton flag identities', () => {
  const {device, graph, props, view} = context();
  try {
    const status = {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      invalidCount: view('uint32', 1),
      overflow: view('uint32', 1)
    };
    const result = addSpatialRateRecipe(graph, {
      events: view('float32', 2),
      populations: view('float32', 2),
      weights: {...props.weights, status}
    });
    expect(result.status.stages[0].status).toMatchObject(status);
    expect(result.status.stages[0].status.overflow).toBe(status.overflow);
  } finally {
    device.destroy();
  }
});
it('B5 captures output getters once and accepts inherited zonal props', () => {
  const {device, graph, view} = context();
  try {
    const output = view('float32', 2);
    let reads = 0;
    const result = addExcessRiskRecipe(graph, {
      events: view('float32', 2),
      populations: view('float32', 2),
      get output() {
        reads++;
        return reads === 1 ? output : view('float32', 2);
      }
    });
    expect(reads).toBe(1);
    expect(result.outputs.excessRisk).toBe(output);
    const zonal = Object.create({
      values: view('float32', 2),
      source: {kind: 'zones', zones: view('uint32', 2), zoneCount: 1}
    });
    expect(() => addZonalDistributionRecipe(graph, zonal)).not.toThrow();
  } finally {
    device.destroy();
  }
});
it('S1 rejects zero-length slice loss and counts invalid live timestamp rows', () => {
  const {device, graph, props, view} = context();
  try {
    expect(() =>
      addStopValueCubeRecipe(graph, {...props, sliceStatuses: [{overflow: view('uint32', 0)}]})
    ).toThrow();
  } finally {
    device.destroy();
  }
  const next = context();
  try {
    const result = addStopValueCubeRecipe(next.graph, next.props);
    expect(result.status.invalidCount).toBeDefined();
    const memory = createCPUGraphMemory();
    memory.write(next.props.stops.status.count, Uint32Array.of(1));
    memory.write(next.props.stops.startRows, Uint32Array.of(99, 99));
    const classify = next.nodes().find(node => node.id === 'stop-value-cube-invalid')!;
    executeCommandNodesOnCPU([classify], memory);
    const rowOutput = (
      classify.resources as {
        buffer: import('@luma.gl/graph/gpu-core').GraphDataView<'uint32'>;
        usage: string;
      }[]
    ).find(resource => resource.usage === 'storage-read-write')!.buffer;
    expect(Array.from(memory.readUint32(rowOutput))).toEqual([1, 0]);
  } finally {
    next.device.destroy();
  }
});
function choroplethContext() {
  const owner = context();
  const {view} = owner;
  return {
    ...owner,
    input: {
      points: {kind: 'points' as const, positions: view('float32x2', 3)},
      polygons: {
        kind: 'polygons' as const,
        positions: view('float32x2', 4),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2),
        candidateCapacity: 8
      },
      values: view('float32', 3),
      output: {featureValues: view('float32', 1)}
    }
  };
}
it.each([
  {backend: 'grp'},
  {statistic: 'total'}
])('B9 rejects invalid recipe options %j', options => {
  const {device, graph, input} = choroplethContext();
  try {
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(graph, {...input, statistic: 'sum', ...options} as never)
    ).toThrow(/backend|statistic/);
  } finally {
    device.destroy();
  }
});
it('earlier color destination requires color settings', () => {
  const {device, graph, input, view} = choroplethContext();
  try {
    expect(() =>
      addPointsInPolygonsChoroplethRecipe(graph, {
        ...input,
        output: {color: {colors: view('uint32', 1)}}
      })
    ).toThrow(/color/);
  } finally {
    device.destroy();
  }
});
it.each(['sum', 'mean'] as const)('earlier CPU %s preserves cancellation residual', statistic => {
  const result = computePointsInPolygonsChoroplethOnCPU(
    {
      points: {kind: 'points', positions: Float32Array.of(0.25, 0.25, 0.5, 0.5, 0.75, 0.75)},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 4)
      },
      values: Float32Array.of(2 ** 55, 1, -(2 ** 55))
    },
    {statistic, candidateCapacity: 8}
  );
  expect(Array.from(result.counts)).toEqual([3]);
  const expected = statistic === 'sum' ? 1 : Math.fround(1 / 3); // Exact cancellation leaves one, divided by three for mean.
  expect(Number.isFinite(result.featureValues[0])).toBe(true);
  expect(result.featureValues[0]).toBe(expected);
});
