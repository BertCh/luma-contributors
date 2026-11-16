// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(graph: GPUCommandGraph, triangles = 2) {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    triangles: createTransientView(graph, 'triangles', 'uint32x3', triangles),
    triangleCount: createTransientView(graph, 'triangle-count', 'uint32', 1),
    clipBounds: createTransientView(graph, 'clip', 'float32x4', 1),
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      siteIds: createTransientView(graph, 'sites', 'uint32x2', 1),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
}

it('accepts segment/site fields packed across separate passes and rejects co-bound prefixes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'packed', 'uint32', 7);
  props.output.segments = graph.createDataView(packed.buffer, {format: 'float32x4', length: 1});
  props.output.siteIds = graph.createDataView(packed.buffer, {
    format: 'uint32x2',
    byteOffset: 16,
    length: 1
  });
  expect(() => graph.add(new GPUVoronoiDiagram(props).getCommandNodes(graph))).not.toThrow();
  expect(
    () =>
      new GPUVoronoiDiagram({
        ...props,
        output: {
          ...props.output,
          status: {
            ...props.output.status,
            requiredCount: graph.createDataView(packed.buffer, {
              format: 'uint32',
              byteOffset: 24,
              length: 1
            })
          }
        }
      })
  ).toThrow(/aligned binding.*must not overlap/);
  device.destroy();
});

it('preflights candidate endpoints and STORAGE usage before any declarations', () => {
  for (const failure of ['scratch', 'usage'] as const) {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 512});
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 11);
    if (failure === 'usage') {
      const buffer = graph.createTransientBuffer({
        id: 'non-storage',
        byteLength: 16,
        usage: Buffer.COPY_SRC | Buffer.COPY_DST
      });
      props.clipBounds = graph.createDataView(buffer, {format: 'float32x4', length: 1});
    }
    const contributor = new GPUVoronoiDiagram(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      failure === 'scratch'
        ? /candidate segment scratch needs 528 bytes/
        : /buffer usage must include Buffer.STORAGE/
    );
    expect(allocate).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    device.destroy();
  }
});
