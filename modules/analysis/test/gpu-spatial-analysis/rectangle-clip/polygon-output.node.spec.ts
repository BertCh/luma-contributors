// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {
  GPURectangleClip,
  getGPURectangleClipParameterValues
} from '../../../src/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
function createProps(graph: GPUCommandGraph, features = 1) {
  return {
    id: 'polygon-clip',
    geometryType: 'polygons' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 7),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 3),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 3),
    featureOffsets: createTransientView(graph, 'features', 'uint32', features + 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      positions: createTransientView(graph, 'result', 'float32x2', 16),
      lineOffsets: createTransientView(graph, 'offsets', 'uint32', 3),
      status: {
        count: createTransientView(graph, 'ring-count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        vertexCount: createTransientView(graph, 'vertex-count', 'uint32', 1)
      },
      polygonFeatures: {
        sourceRows: createTransientView(graph, 'sources', 'uint32', features),
        count: createTransientView(graph, 'feature-count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required-features', 'uint32', 1)
      }
    }
  };
}

it.each(
  [1, 2].flatMap(features =>
    (['canonical', 'legacy', 'both'] as const).map(spelling => ({features, spelling}))
  )
)('publishes a standard polygon tree with $features features ($spelling status)', ({
  features,
  spelling
}) => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, features);
  if (spelling === 'canonical') {
    const packed = createTransientView(graph, 'packed-positions-and-feature-count', 'uint32', 15);
    props.positions = graph.createDataView(packed.buffer, {format: 'float32x2', length: 7});
    props.output.polygonFeatures.count = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 56
    });
  }
  const {count, requiredCount, sourceRows} = props.output.polygonFeatures;
  const status = {
    count: graph.createDataView(count.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: count.byteOffset
    }),
    requiredCount: graph.createDataView(requiredCount.buffer, {format: 'uint32', length: 1})
  };
  const polygonFeatures =
    spelling === 'legacy'
      ? props.output.polygonFeatures
      : {sourceRows, status, ...(spelling === 'both' ? {count, requiredCount} : {})};
  const contributor = new GPURectangleClip({...props, output: {...props.output, polygonFeatures}});
  const result = contributor.getPolygonOutput();
  expect(result.status.count).toBe(spelling === 'legacy' ? count : status.count);
  expect(result.status.requiredCount).toBe(
    spelling === 'legacy' ? requiredCount : status.requiredCount
  );
  expect(result.geometry.featureOffsets).toBe(props.featureOffsets);
  expect(result.geometry.polygonOffsets).toBe(props.polygonOffsets);
  expect(result.geometry.ringOffsets).toBe(props.output.lineOffsets);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const memory = createCPUGraphMemory();
  memory.write(
    props.positions,
    new Float32Array([-1, -1, 3, -1, 3, 3, -1, 3, 10, 10, 11, 10, 10, 11])
  );
  memory.write(props.ringOffsets, new Uint32Array([0, 4, 7]));
  memory.write(props.polygonOffsets, new Uint32Array([0, 1, 2]));
  memory.write(props.featureOffsets, new Uint32Array(features === 1 ? [0, 2] : [0, 2, 2]));
  memory.write(
    props.parameters,
    getGPURectangleClipParameterValues({minX: 0, minY: 0, maxX: 2, maxY: 2})
  );
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.lineOffsets))).toEqual([0, 4, 4]);
  const coordinates = Array.from(memory.readFloat32(props.output.positions, 8));
  const points = Array.from({length: 4}, (_, index) => coordinates.slice(index * 2, index * 2 + 2));
  const start = points.findIndex(point => point[0] === 0 && point[1] === 0);
  expect([...points.slice(start), ...points.slice(0, start)]).toEqual([
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2]
  ]);
  expect(Array.from(memory.readUint32(props.output.polygonFeatures.sourceRows))).toEqual(
    features === 1 ? [0] : [0, 1]
  );
  expect(Array.from(memory.readFloat32(props.positions))).toEqual([
    -1, -1, 3, -1, 3, 3, -1, 3, 10, 10, 11, 10, 10, 11
  ]);
  expect(memory.readUint32(count)[0]).toBe(features);
  expect(memory.readUint32(requiredCount)[0]).toBe(features);
  expect(memory.readUint32(props.output.status.count)[0]).toBe(2);
  expect(memory.readUint32(props.output.status.overflow)[0]).toBe(0);
  for (const {source} of getGPUCommandNodeKernelSources(nodes))
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
});

it('accepts packed status across separate passes and rejects a same-pass aligned overlap', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const shared = createTransientView(graph, 'packed-status', 'uint32', 65);
  const count = graph.createDataView(shared.buffer, {format: 'uint32', length: 1});
  const overflow = graph.createDataView(shared.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        output: {...props.output, status: {...props.output.status, count, overflow}}
      })
  ).not.toThrow();
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        output: {...props.output, status: {...props.output.status, overflow, vertexCount: count}}
      })
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('preflights scratch and actual caller bindings before declaring resources', () => {
  const constrained = createNullWebGPUDevice();
  Object.defineProperty(constrained.limits, 'maxStorageBufferBindingSize', {value: 64});
  const graph = new GPUCommandGraph(constrained);
  const contributor = new GPURectangleClip(createProps(graph));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 128 bytes/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    constrained.destroy();
  }
});

it('accepts line-mode status words packed across the publish and tail dispatches', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'line-packed-status', 'uint32', 65);
  const count = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const overflow = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        geometryType: 'lines',
        lineOffsets: props.ringOffsets,
        output: {
          ...props.output,
          polygonFeatures: undefined,
          status: {...props.output.status, count, overflow}
        }
      })
  ).not.toThrow();
});

it.each([
  'featureOffsets',
  'polygonOffsets'
] as const)('protects retained %s from provenance publication', name => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 2);
  const sourceRows = graph.createDataView(props[name].buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: 4
  });
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        output: {...props.output, polygonFeatures: {...props.output.polygonFeatures, sourceRows}}
      })
  ).toThrow(/share|overlap|disjoint/);
});

it('accepts provenance packed beyond retained hierarchy in the same physical buffer', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 2);
  const packed = createTransientView(graph, 'retained-and-output', 'uint32', 128);
  const featureOffsets = graph.createDataView(packed.buffer, {format: 'uint32', length: 3});
  const sourceRows = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: 256
  });
  expect(() =>
    graph.add(
      new GPURectangleClip({
        ...props,
        featureOffsets,
        output: {...props.output, polygonFeatures: {...props.output.polygonFeatures, sourceRows}}
      })
    )
  ).not.toThrow();
});

it.each([
  'count',
  'requiredCount'
] as const)('rejects conflicting feature %s aliases at construction', name => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const features = props.output.polygonFeatures;
  const replacement = createTransientView(graph, 'conflicting-status', 'uint32', 1);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        output: {
          ...props.output,
          polygonFeatures: {
            ...features,
            status: {
              count: features.count,
              requiredCount: features.requiredCount,
              [name]: replacement
            }
          }
        }
      })
  ).toThrow(new RegExp(`^polygon-clip output.polygonFeatures.status.${name}.*deprecated alias`));
  expect(allocate).not.toHaveBeenCalled();
});

it('validates the canonical feature count length at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const count = createTransientView(graph, 'wrong-count-length', 'uint32', 2);
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        output: {
          ...props.output,
          polygonFeatures: {
            sourceRows: props.output.polygonFeatures.sourceRows,
            status: {count}
          }
        }
      })
  ).toThrow(/^polygon-clip.*count.*one scalar/);
});

it('preflights STORAGE usage on the nested canonical status before graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const diagnostic = graph.createDataView(
    graph.createTransientBuffer({
      id: 'non-storage-diagnostic',
      byteLength: 4,
      usage: Buffer.COPY_DST
    }),
    {format: 'uint32', length: 1}
  );
  const contributor = new GPURectangleClip({
    ...props,
    output: {
      ...props.output,
      polygonFeatures: {
        sourceRows: props.output.polygonFeatures.sourceRows,
        status: {count: diagnostic}
      }
    }
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('allows a separate-pass feature count beside input positions and protects their logical words', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'positions-and-feature-count', 'uint32', 15);
  const positions = graph.createDataView(packed.buffer, {format: 'float32x2', length: 7});
  const count = graph.createDataView(packed.buffer, {format: 'uint32', length: 1, byteOffset: 56});
  const featureOutput = {sourceRows: props.output.polygonFeatures.sourceRows, status: {count}};
  expect(() =>
    graph.add(
      new GPURectangleClip({
        ...props,
        positions,
        output: {
          ...props.output,
          polygonFeatures: featureOutput
        }
      })
    )
  ).not.toThrow();
  const overlapping = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 52
  });
  expect(
    () =>
      new GPURectangleClip({
        ...props,
        positions,
        output: {
          ...props.output,
          polygonFeatures: {...featureOutput, status: {count: overlapping}}
        }
      })
  ).toThrow(/logical byte ranges/);
});

it('rejects disjoint read/write ranges of one physical buffer in a polygon emission dispatch', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'parameters-and-output', 'uint32', 96);
  const parameters = graph.createDataView(packed.buffer, {format: 'float32', length: 4});
  const positions = graph.createDataView(packed.buffer, {
    format: 'float32x2',
    length: 16,
    byteOffset: 256
  });
  expect(
    () => new GPURectangleClip({...props, parameters, output: {...props.output, positions}})
  ).toThrow(/outputs must not share buffers with inputs/);
});
