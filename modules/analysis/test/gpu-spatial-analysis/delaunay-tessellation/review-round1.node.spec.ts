// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUDelaunayTessellation,
  type GPUDelaunayTessellationProps
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram';

function createDelaunayProps(graph: GPUCommandGraph, prefix: string): GPUDelaunayTessellationProps {
  return {
    positions: createTransientView(graph, `${prefix}-positions`, 'float32x2', 3),
    output: {
      triangles: createTransientView(graph, `${prefix}-triangles`, 'uint32x3', 1),
      status: {
        count: createTransientView(graph, `${prefix}-count`, 'uint32', 1),
        requiredCount: createTransientView(graph, `${prefix}-required`, 'uint32', 1),
        overflow: createTransientView(graph, `${prefix}-overflow`, 'uint32', 1)
      }
    }
  };
}

it('B1 rejects an inherited longitude-latitude declaration instead of losing it in a spread', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let reads = 0;
  const props = Object.setPrototypeOf(createDelaunayProps(graph, 'inherited'), {
    get spatialContext() {
      reads++;
      return {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'};
    }
  });
  try {
    expect(() => new GPUDelaunayTessellation(props)).toThrow(/longitude-latitude/);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('B1 captures output and nested group maps once before validation and lowering', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const first = createDelaunayProps(graph, 'first');
  const second = createDelaunayProps(graph, 'second');
  const groupIds = createTransientView(graph, 'groups', 'uint32', 3);
  const triangleCounts = createTransientView(graph, 'group-counts', 'uint32', 1);
  const converged = createTransientView(graph, 'group-converged', 'uint32', 1);
  const reads = {output: 0, groups: 0, status: 0, converged: 0};
  type OutputGroups = NonNullable<GPUDelaunayTessellationProps['output']['groups']>;
  const groupStatus: OutputGroups['status'] = Object.create(null, {
    converged: {
      get() {
        reads.converged++;
        return converged;
      }
    }
  });
  const outputGroups: OutputGroups = Object.create(null, {
    triangleCounts: {
      get() {
        return triangleCounts;
      }
    },
    status: {
      get() {
        reads.status++;
        return groupStatus;
      }
    }
  });
  const output: GPUDelaunayTessellationProps['output'] = Object.create(null, {
    triangles: {
      get() {
        return first.output.triangles;
      }
    },
    status: {
      get() {
        return first.output.status;
      }
    },
    groups: {
      get() {
        reads.groups++;
        return outputGroups;
      }
    }
  });
  const props: GPUDelaunayTessellationProps = {
    positions: first.positions,
    groups: {ids: groupIds, count: 1},
    get output() {
      return reads.output++ === 0
        ? output
        : {...second.output, groups: {triangleCounts, status: {converged}}};
    }
  };
  try {
    const contributor = new GPUDelaunayTessellation(props);
    const nodes = contributor.getCommandNodes(graph);
    expect(contributor.props.output.triangles === first.output.triangles).toBe(true);
    expect(contributor.props.output.status.count === first.output.status.count).toBe(true);
    expect(contributor.props.output.groups?.triangleCounts === triangleCounts).toBe(true);
    expect(contributor.props.output.groups?.status.converged === converged).toBe(true);
    expect(
      nodes.some(node =>
        getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
          binding => binding.view === first.output.triangles && binding.access === 'read_write'
        )
      )
    ).toBe(true);
    expect(reads).toEqual({output: 1, groups: 1, status: 1, converged: 1});
  } finally {
    device.destroy();
  }
});

it.each([
  'count',
  'overflow'
] as const)('N1 rejects physical sharing between Voronoi requiredCount and %s', field => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const packed = createTransientView(graph, 'packed', 'uint32', 65);
  const requiredCount = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const shared = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  const props = createDelaunayProps(graph, 'voronoi');
  try {
    expect(
      () =>
        new GPUVoronoiDiagram({
          positions: props.positions,
          triangles: props.output.triangles,
          triangleCount: createTransientView(graph, 'requested', 'uint32', 1),
          clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
          output: {
            segments: createTransientView(graph, 'segments', 'float32x4', 3),
            siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
            status: {...props.output.status, requiredCount, [field]: shared}
          }
        })
    ).toThrow(/must not share buffers|physical buffer/);
  } finally {
    device.destroy();
  }
});
