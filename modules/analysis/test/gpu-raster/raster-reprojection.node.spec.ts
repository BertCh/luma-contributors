// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';
import {expect, it, vi} from 'vitest';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {getRasterBufferBandNodes} from '../../src/raster-analysis/raster-band-utils';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

function createCase(width = 17, height = 3) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const count = width * height;
  const grid = {
    width,
    height,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const props = {
    id: 'reprojection',
    sourceGrid: grid,
    destinationGrid: grid,
    input: {
      id: 'input',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: createTransientView(graph, 'input', 'float32', count)
      }
    },
    output: {
      values: createTransientView(graph, 'output', 'float32', count),
      validity: createTransientView(graph, 'validity', 'uint32', count)
    }
  };
  return {graph, props, count};
}

it.each([1, 51, 1025])('removes full-destination scratch at %i cells (LUM-26)', count => {
  // Measure logical transient bytes, excluding caller-owned views and projection parameters.
  // Reconstruct the previous allocation plan: canonical band + grid + projected positions/validity.
  const previous = createCase(count, 1);
  const previousAllocations = vi.spyOn(previous.graph, 'createTransientBuffer');
  getRasterBufferBandNodes(
    previous.graph,
    'previous',
    previous.props.input,
    previous.props.sourceGrid,
    true
  );
  createTransientView(previous.graph, 'destination', 'float32x2', count);
  createTransientView(previous.graph, 'projected', 'float32x2', count);
  createTransientView(previous.graph, 'projection-validity', 'uint32', count);
  const previousBytes = previousAllocations.mock.results.reduce(
    (total, result) => total + result.value.byteLength,
    0
  );

  const {graph, props} = createCase(count, 1);
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  const contributor = new GPURasterReprojection({
    ...props,
    inverseProjection: compileProjectionProgram({
      precision: 'local-f32',
      operations: [{type: 'affine', scale: [1, 1], offset: [0.25, 0.25]}]
    })
  });
  try {
    const nodes = contributor.getCommandNodes(graph);
    const currentBytes = allocations.mock.results.reduce(
      (total, result) => total + result.value.byteLength,
      0
    );
    expect(previousBytes).toBe(28 * count);
    expect(currentBytes).toBe(8 * count); // Only source-band canonicalization remains.
    expect(previousBytes - currentBytes).toBe(20 * count);
    expect(
      nodes
        .map(node => node.id)
        .filter(id => id.startsWith('reprojection-') && !id.includes('elevation'))
    ).toEqual(['reprojection-sample']);
    const kernels = getParsedKernelSources(nodes);
    const sampler = kernels.get('reprojection-sample')!;
    expect(sampler).toContain('projection_raster_project');
    expect(getStorageBindingCount(sampler)).toBe(6);
    expect(sampler).not.toContain('@binding(auto)');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/^reprojection .*once/);
  } finally {
    contributor.destroy();
    previous.graph.device.destroy();
    graph.device.destroy();
  }
});

it('same-CRS sampling has no destination scratch or grid pass', () => {
  const {graph, props} = createCase();
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  const nodes = new GPURasterReprojection(props).getCommandNodes(graph);
  expect(allocations.mock.results.map(result => result.value.byteLength)).toEqual([204, 204]);
  const kernels = getParsedKernelSources(nodes);
  expect([...kernels.keys()].filter(id => !id.includes('elevation'))).toEqual([
    'reprojection-sample'
  ]);
  expect(getStorageBindingCount(kernels.get('reprojection-sample')!)).toBe(5);
  graph.device.destroy();
});

it('accepts double-single projection output and parses the inline origin subtraction', () => {
  const {graph, props} = createCase();
  const contributor = new GPURasterReprojection({
    ...props,
    inverseProjection: compileProjectionProgram({
      precision: 'double-single',
      operations: [{type: 'affine', scale: [1, 1], offset: [500000.3, 4600000.1]}]
    })
  });
  try {
    const sampler = getParsedKernelSources(contributor.getCommandNodes(graph)).get(
      'reprojection-sample'
    )!;
    expect(sampler).toContain('sub_fp64(projected.position.xy');
    expect(sampler).toContain('sub_fp64(projected.position.zw');
    expect(getStorageBindingCount(sampler)).toBe(6);
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});

it('validates fixed layout requirements at construction', () => {
  const {graph, props, count} = createCase();
  try {
    expect(
      () =>
        new GPURasterReprojection({
          ...props,
          sourceGrid: {...props.sourceGrid, affine: [1, 2, 0, 2, 4, 0]}
        })
    ).toThrow(/^reprojection sourceGrid.affine .*invertible/);
    expect(
      () =>
        new GPURasterReprojection({
          ...props,
          output: {
            ...props.output,
            values: createTransientView(graph, 'short', 'float32', count - 1)
          }
        })
    ).toThrow(/^reprojection output.values .*at least 51/);
    expect(
      () =>
        new GPURasterReprojection({
          ...props,
          inverseProjection: compileProjectionProgram(
            {precision: 'local-f32', operations: []},
            {inputFormat: 'float32x4'}
          )
        })
    ).toThrow(/^reprojection inverseProjection.inputFormat must be 'float32x2'/);
  } finally {
    graph.device.destroy();
  }
});

it.each([
  203, 204
])('checks source scratch against the %i-byte device binding limit before creating nodes', limit => {
  const {graph, props} = createCase();
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxStorageBufferBindingSize: limit}
  });
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  const contributor = new GPURasterReprojection(props);
  try {
    if (limit < 204) {
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /^reprojection sourceGrid canonical band needs 204 bytes in one binding \(limit 203\)/
      );
      expect(allocations).not.toHaveBeenCalled();
    } else {
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    }
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});

it('checks projection parameter storage before allocating source scratch', () => {
  const {graph, props} = createCase(1, 1);
  const inverseProjection = compileProjectionProgram({
    precision: 'local-f32',
    operations: [{type: 'affine', scale: [1, 1], offset: [0, 0]}]
  });
  const limit = inverseProjection.packParameters().byteLength - 1;
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxStorageBufferBindingSize: limit}
  });
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  const contributor = new GPURasterReprojection({...props, inverseProjection});
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^reprojection inverseProjection parameters needs .*limit/
    );
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});

it('parses a callable with an invalid projection domain', () => {
  const {graph, props} = createCase();
  const contributor = new GPURasterReprojection({
    ...props,
    inverseProjection: compileProjectionProgram({
      precision: 'local-f32',
      operations: [{type: 'web-mercator', arithmetic: 'float32', radius: 1}]
    })
  });
  try {
    const sampler = getParsedKernelSources(contributor.getCommandNodes(graph)).get(
      'reprojection-sample'
    )!;
    expect(sampler).toContain('projected.valid!=0u');
    expect(getStorageBindingCount(sampler)).toBe(6);
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});

it('validates double-single origin representation at construction', () => {
  const {graph, props} = createCase();
  try {
    expect(
      () =>
        new GPURasterReprojection({
          ...props,
          sourceGrid: {...props.sourceGrid, affine: [1, 0, 1e40, 0, 1, 0]},
          inverseProjection: compileProjectionProgram({precision: 'double-single', operations: []})
        })
    ).toThrow(/^reprojection sourceGrid.affine\[2\] must be a finite number/);
  } finally {
    graph.device.destroy();
  }
});

it('checks the affine parameter binding before allocating source scratch', () => {
  const {graph, props} = createCase(1, 1);
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxStorageBufferBindingSize: 79}
  });
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  const contributor = new GPURasterReprojection(props);
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^reprojection affine parameters needs 80 bytes in one binding \(limit 79\)/
    );
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});
