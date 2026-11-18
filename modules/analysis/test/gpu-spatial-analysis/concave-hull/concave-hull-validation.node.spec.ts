// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUConcaveHull,
  GPUGroupConcaveHull
} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createConcaveHullTestOutput,
  createConcaveHullValidationProps as createProps
} from './concave-hull-validation-cases';
import {validateGraphChunkBindingRanges} from '../../../src/analysis-core/kernels/graph-chunk-dispatch';

it('headless output fixture gives empty vector views a full physical binding row', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (const format of ['uint32', 'uint32x2', 'uint32x3'] as const) {
      const output = createConcaveHullTestOutput(graph, device, `empty-${format}`, format, 0);
      try {
        expect(output.view.length).toBe(0);
        expect(() =>
          validateGraphChunkBindingRanges(graph, 'empty-hull-output', {output: output.view})
        ).not.toThrow();
      } finally {
        output.buffer.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});

it('validates grouped and single radii before a nonfinite override reaches a pipeline', () => {
  for (const grouped of [false, true]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const groupIds = createTransientView(graph, 'groups', 'uint32', 4);
    for (const maximumCircumradius of [NaN, Infinity, 1e40, 0, -1, 1e-100]) {
      expect(() =>
        grouped
          ? new GPUGroupConcaveHull({...props, groupIds, groupCount: 1, maximumCircumradius})
          : new GPUConcaveHull({...props, maximumCircumradius})
      ).toThrow(/hull maximumCircumradius/);
    }
    const contributor = grouped
      ? new GPUGroupConcaveHull({...props, groupIds, groupCount: 1})
      : new GPUConcaveHull(props);
    expect(() => graph.add(contributor.getCommandNodes(graph))).not.toThrow();
    device.destroy();
  }
});

it('preflights nested grouped topology before allocating any enclosing resources', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 1024});
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 1, 100);
  const contributor = new GPUGroupConcaveHull({
    ...props,
    groupIds: createTransientView(graph, 'groups', 'uint32', 1),
    groupCount: 100
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /triangle topology scratch needs 1744 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});

it('preflights the actual boundary binding before declaring scalar or adjacency scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUConcaveHull(createProps(graph, 2_796_206));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 134217768 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});

it('preserves named radius and output-limit diagnostics on the browser validation fixtures', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 6);
    expect(() => new GPUConcaveHull(props)).not.toThrow();
    expect(() => new GPUConcaveHull({...props, maximumCircumradius: 1e39})).toThrow(
      /^concave-hull maximumCircumradius must be a finite number in \[0, Infinity\]/
    );
    expect(() => new GPUConcaveHull({...props, maximumCircumradius: 0})).toThrow(
      /^concave-hull maximumCircumradius must be greater than 0/
    );
    const undersizedTriangles = graph.createDataView(props.output.triangles.buffer, {
      format: 'uint32x3',
      length: 6
    });
    expect(
      () =>
        new GPUConcaveHull({
          ...props,
          output: {...props.output, triangles: undersizedTriangles}
        })
    ).toThrow(/^concave-hull output.triangles must have 7 rows/);
  } finally {
    device.destroy();
  }
});

it('preserves group-count and convergence diagnostics with valid packed views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 6, 2);
    const groupIds = createTransientView(graph, 'groups', 'uint32', 6);
    expect(() => new GPUGroupConcaveHull({...props, groupIds, groupCount: 2})).not.toThrow();
    expect(() => new GPUGroupConcaveHull({...props, groupIds, groupCount: 0})).toThrow(
      /^group-concave-hull groupCount must be an integer in \[1, 4294967295\]/
    );
    const converged = graph.createDataView(props.output.status.converged.buffer, {
      format: 'uint32',
      length: 1
    });
    expect(
      () =>
        new GPUGroupConcaveHull({
          ...props,
          groupIds,
          groupCount: 2,
          output: {...props.output, status: {converged}}
        })
    ).toThrow(/^group-concave-hull output.status.converged must have at least 2 rows/);
  } finally {
    device.destroy();
  }
});

it('packed grouped hull reserves the global triangle bound instead of multiplying by groupCount', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, 6, 2, 7);
    const contributor = new GPUGroupConcaveHull({
      ...props,
      layout: 'packed',
      groupIds: createTransientView(graph, 'groups', 'uint32', 6),
      groupCount: 2,
      output: {
        ...props.output,
        status: {
          ...props.output.status,
          count: createTransientView(graph, 'count', 'uint32', 1),
          requiredCount: createTransientView(graph, 'required', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        },
        groups: {
          triangleOffsets: createTransientView(graph, 'triangle-offsets', 'uint32', 2),
          triangleCounts: createTransientView(graph, 'triangle-counts', 'uint32', 2),
          edgeOffsets: createTransientView(graph, 'edge-offsets', 'uint32', 2),
          edgeCounts: createTransientView(graph, 'edge-counts', 'uint32', 2)
        }
      }
    });
    expect(() => graph.add(contributor.getCommandNodes(graph))).not.toThrow();
  } finally {
    device.destroy();
  }
});
