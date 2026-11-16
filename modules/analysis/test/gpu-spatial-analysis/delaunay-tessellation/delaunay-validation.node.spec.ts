// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUDelaunayTessellation,
  type GPUDelaunayTessellationProps
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(graph: GPUCommandGraph, pointCount = 8): GPUDelaunayTessellationProps {
  return {
    id: 'delaunay',
    positions: createTransientView(graph, 'positions', 'float32x2', pointCount),
    output: {
      triangles: createTransientView(graph, 'triangles', 'uint32x3', 1),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1)
      }
    }
  };
}

it('rejects overlapping status fields and aligned writable prefixes at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUDelaunayTessellation({
        ...props,
        output: {
          ...props.output,
          status: {...props.output.status, overflow: props.output.status.count}
        }
      })
  ).toThrow(/must not overlap/);
  const backing = createTransientView(graph, 'backing', 'uint32', 65);
  const count = graph.createDataView(backing.buffer, {format: 'uint32', byteOffset: 0, length: 1});
  const overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  expect(
    () =>
      new GPUDelaunayTessellation({
        ...props,
        output: {...props.output, status: {...props.output.status, count, overflow}}
      })
  ).toThrow(/aligned binding.*must not overlap/);
  // Aligned, non-overlapping reservations in one buffer are valid.
  expect(
    () =>
      new GPUDelaunayTessellation({
        ...props,
        output: {
          ...props.output,
          status: {
            ...props.output.status,
            count,
            overflow: graph.createDataView(backing.buffer, {
              format: 'uint32',
              byteOffset: 256,
              length: 1
            })
          }
        }
      })
  ).not.toThrow();
  device.destroy();
});

it('rejects output-versus-input aliases and unrepresentable derived word addresses', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUDelaunayTessellation({
        ...props,
        output: {
          ...props.output,
          triangles: graph.createDataView(props.positions.buffer, {format: 'uint32x3', length: 1})
        }
      })
  ).toThrow(/outputs must not share buffers with inputs/);
  expect(
    () =>
      new GPUDelaunayTessellation({
        ...props,
        groups: {ids: createTransientView(graph, 'groups', 'uint32', 8), count: 0x100000000}
      })
  ).toThrow(/delaunay groups.count/);
  const hugeProps = createProps(new GPUCommandGraph(device), 536_870_912);
  expect(() => new GPUDelaunayTessellation(hugeProps)).toThrow(/scratch triangle count/);
  device.destroy();
});

it('checks aligned bindings and the largest topology scratch without mutating the graph', () => {
  for (const failure of ['scratch', 'aligned input', 'output'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, failure === 'scratch' ? 4_194_301 : 8);
    if (failure === 'aligned input') {
      const backing = createTransientView(graph, 'large-positions', 'float32x2', 16_777_217);
      props.positions = graph.createDataView(backing.buffer, {
        format: 'float32x2',
        byteOffset: 4,
        length: 16_777_216
      });
    }
    if (failure === 'output')
      props.output.triangles = createTransientView(graph, 'large-output', 'uint32x3', 11_184_811);
    const contributor = new GPUDelaunayTessellation(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 1342177\d+ bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('includes grouped outputs in constructor alias validation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUDelaunayTessellation({
        ...props,
        groups: {ids: createTransientView(graph, 'groups', 'uint32', 8), count: 1},
        output: {
          ...props.output,
          groups: {
            triangleCounts: props.output.status.count,
            status: {converged: createTransientView(graph, 'group-converged', 'uint32', 1)}
          }
        }
      })
  ).toThrow(/must not overlap/);
  device.destroy();
});

it('accepts status packing across dispatches and ignores unused group outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const backing = createTransientView(graph, 'packed-status', 'uint32', 2);
  props.output.status.count = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
  props.output.status.converged = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  props.output.groups = {
    triangleCounts: props.output.status.count,
    status: {converged: props.output.status.count}
  };
  expect(() => graph.add(new GPUDelaunayTessellation(props).getCommandNodes(graph))).not.toThrow();
  device.destroy();
});
