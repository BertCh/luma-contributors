// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';
import {expect, it, vi} from 'vitest';
import {
  GPURasterReprojection,
  getGPURasterReprojectionParameterValues
} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPURasterTileReprojection} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
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

it('B1 emits a representable native INT32_MIN sentinel in both buffer warp gathers', () => {
  for (const tiled of [false, true]) {
    const {graph, props} = createCase(2, 1);
    const input: GPURasterBand = {
      id: 'signed',
      format: 'sint32',
      noDataValue: -2147483648,
      storage: {kind: 'buffer', values: createTransientView(graph, 'signed', 'sint32', 2)}
    };
    const output = {...props.output, labels: createTransientView(graph, 'labels', 'sint32', 2)};
    const contributor = tiled
      ? new GPURasterTileReprojection({
          ...props,
          output,
          method: 'nearest',
          sources: [{pixelBounds: [0, 0, 2, 1], input}]
        })
      : new GPURasterReprojection({...props, output, method: 'nearest', input});
    try {
      const sources = [...getParsedKernelSources(contributor.getCommandNodes(graph)).values()].join(
        '\n'
      );
      expect(sources).toContain('value!=bitcast<i32>(0x80000000u)');
      expect(sources).not.toContain('-2147483648i');
    } finally {
      contributor.destroy();
      graph.device.destroy();
    }
  }
});

it('B3 rejects incompatible tiled projection before allocation and permits a same-ID retry', () => {
  const {graph, props} = createCase(2, 1);
  const sources = [{pixelBounds: [0, 0, 2, 1] as const, input: props.input}];
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(
      () =>
        new GPURasterTileReprojection({
          ...props,
          sources,
          inverseProjection: compileProjectionProgram(
            {precision: 'local-f32', operations: []},
            {inputFormat: 'float32x4'}
          )
        })
    ).toThrow(/inverseProjection.inputFormat.*float32x2/);
    expect(allocations).not.toHaveBeenCalled();
    const retry = new GPURasterTileReprojection({
      ...props,
      sources,
      inverseProjection: compileProjectionProgram(
        {precision: 'local-f32', operations: []},
        {inputFormat: 'float32x2'}
      )
    });
    try {
      expect(() => retry.getCommandNodes(graph)).not.toThrow();
    } finally {
      retry.destroy();
    }
  } finally {
    graph.device.destroy();
  }
});

it.each([
  'inherited',
  'non-enumerable'
])('B5 captures %s rank policy and rejects its typos', location => {
  const {graph, props} = createCase(2, 1);
  const create = (noDataPolicy: string) => {
    const fields = {...props, grid: props.sourceGrid, radius: 1};
    const settings =
      location === 'inherited'
        ? Object.assign(Object.create({noDataPolicy}), fields)
        : Object.defineProperty(fields, 'noDataPolicy', {value: noDataPolicy});
    return new GPURasterRankFilter(settings);
  };
  try {
    expect(create('ignore').props.noDataPolicy).toBe('ignore');
    expect(() => create('typo')).toThrow(/noDataPolicy.*propagate.*strict.*ignore.*typo/);
  } finally {
    graph.device.destroy();
  }
});

it('S4 rejects mismatched per-frame CRS without touching the existing packet', () => {
  const {graph, props} = createCase(2, 1);
  const target = getGPURasterReprojectionParameterValues(props);
  const before = new Uint8Array(target.buffer).slice();
  try {
    expect(() =>
      getGPURasterReprojectionParameterValues(
        {
          sourceGrid: {...props.sourceGrid, coordinateReferenceSystem: {authority: 'EPSG:4326'}},
          destinationGrid: {
            ...props.destinationGrid,
            coordinateReferenceSystem: {authority: 'EPSG:3857'}
          }
        },
        target
      )
    ).toThrow(/coordinate reference system/);
    expect(new Uint8Array(target.buffer)).toEqual(before);
  } finally {
    graph.device.destroy();
  }
});

it('EARLIER affine packing retains every byte after a nonrepresentable origin update', () => {
  const {graph, props} = createCase(2, 1);
  const target = getGPURasterReprojectionParameterValues(props);
  const before = new Uint8Array(target.buffer).slice();
  try {
    expect(() =>
      getGPURasterReprojectionParameterValues(
        {...props, destinationGrid: {...props.destinationGrid, affine: [1, 0, 1e60, 0, 1, 0]}},
        target
      )
    ).toThrow(/finite.*float32/);
    expect(new Uint8Array(target.buffer)).toEqual(before);
  } finally {
    graph.device.destroy();
  }
});

it('NAM-05 rejects unknown warp/rank options at construction and keeps strict semantics', () => {
  const {graph, props} = createCase(2, 1);
  try {
    for (const tiled of [false, true]) {
      const create = (overrides: object) =>
        tiled
          ? new GPURasterTileReprojection({
              ...props,
              sources: [{pixelBounds: [0, 0, 2, 1], input: props.input}],
              ...overrides
            })
          : new GPURasterReprojection({...props, ...overrides});
      expect(() => create({method: 'near'})).toThrow(/method.*nearest.*bilinear.*near/);
      expect(() => create({noDataPolicy: 'ignore'})).toThrow(
        /noDataPolicy.*propagate.*strict.*renormalize/
      );
      expect(() => create({noDataPolicy: 'propagate'})).not.toThrow();
      expect(() => create({noDataPolicy: 'strict'})).not.toThrow();
    }
    expect(
      () =>
        new GPURasterRankFilter({
          ...props,
          grid: props.sourceGrid,
          radius: 1,
          borderMode: 'reflect' as 'clamp'
        })
    ).toThrow(/borderMode.*clamp.*nodata.*reflect/);
    expect(
      () =>
        new GPURasterRankFilter({
          ...props,
          grid: props.sourceGrid,
          radius: 1,
          noDataPolicy: 'renormalize' as 'ignore'
        })
    ).toThrow(/noDataPolicy.*propagate.*strict.*ignore/);
  } finally {
    graph.device.destroy();
  }
});

it('URT-07 accepts reordered CRS fields and rejects mismatches before any tiled allocation', () => {
  const {graph, props} = createCase(2, 1);
  const sourceGrid = {
    ...props.sourceGrid,
    coordinateReferenceSystem: {
      authority: 'EPSG:4326',
      wellKnownText: 'same',
      projectionJson: {axes: ['x', 'y'], id: {code: 4326, authority: 'EPSG'}}
    }
  };
  const destinationGrid = {
    ...sourceGrid,
    coordinateReferenceSystem: {
      projectionJson: {id: {authority: 'EPSG', code: 4326}, axes: ['x', 'y']},
      wellKnownText: 'same',
      authority: 'EPSG:4326'
    }
  };
  const sources = [{pixelBounds: [0, 0, 2, 1] as const, input: props.input}];
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => new GPURasterReprojection({...props, sourceGrid, destinationGrid})).not.toThrow();
    expect(
      () =>
        new GPURasterTileReprojection({
          ...props,
          sourceGrid,
          destinationGrid: {
            ...destinationGrid,
            coordinateReferenceSystem: {authority: 'EPSG:3857'}
          },
          sources
        })
    ).toThrow(/coordinate reference system/);
    expect(allocations).not.toHaveBeenCalled();
    const retry = new GPURasterTileReprojection({...props, sourceGrid, destinationGrid, sources});
    expect(() => retry.getCommandNodes(graph)).not.toThrow();
    retry.destroy();
    expect(
      () =>
        new GPURasterReprojection({
          ...props,
          sourceGrid,
          destinationGrid: {
            ...destinationGrid,
            coordinateReferenceSystem: {
              ...destinationGrid.coordinateReferenceSystem,
              projectionJson: {
                ...destinationGrid.coordinateReferenceSystem.projectionJson,
                axes: ['y', 'x']
              }
            }
          }
        })
    ).toThrow(/coordinate reference system/);
  } finally {
    graph.device.destroy();
  }
});

it('LIF-04 captures both grids and output maps before emitting either sampler', () => {
  for (const tiled of [false, true]) {
    const {graph, props} = createCase(2, 1);
    props.destinationGrid = {...props.sourceGrid};
    const originalOutput = props.output.values;
    const contributor = tiled
      ? new GPURasterTileReprojection({
          ...props,
          sources: [{pixelBounds: [0, 0, 2, 1], input: props.input}]
        })
      : new GPURasterReprojection(props);
    props.destinationGrid.width = 1;
    props.output.values = createTransientView(graph, 'replacement', 'float32', 2);
    try {
      const nodes = contributor.getCommandNodes(graph);
      const samplers = [...getParsedKernelSources(nodes).values()].filter(source =>
        source.includes('let destinationColumn=')
      );
      expect(samplers).toHaveLength(1);
      expect(samplers[0]).toContain('index%2u');
      expect(
        nodes.some(node =>
          node.resources?.some(
            resource => 'buffer' in resource && resource.buffer === originalOutput
          )
        )
      ).toBe(true);
    } finally {
      contributor.destroy();
      graph.device.destroy();
    }
  }
});

it.each([
  'uint32',
  'sint32'
] as const)('URT-04 keeps native %s labels until the final output', format => {
  const {graph, props} = createCase(2, 1);
  const labels = createTransientView(graph, 'labels', format, 2);
  const contributor = new GPURasterReprojection({
    ...props,
    method: 'nearest',
    input: {
      ...props.input,
      format,
      noDataValue: 16777217,
      storage: {kind: 'buffer', values: createTransientView(graph, 'native', format, 2)}
    } as GPURasterBand,
    output: {...props.output, labels}
  });
  try {
    const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.get('reprojection-sample')).toContain('labels[labelsOffset+index]=label');
    expect(kernels.get('reprojection-labels')).toContain(
      `16777217${format === 'uint32' ? 'u' : 'i'}`
    );
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});

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
    for (const {source} of kernels.getConfigurations('reprojection-sample'))
      expect(getStorageBindingCount(source)).toBe(6);
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
  for (const {source} of kernels.getConfigurations('reprojection-sample'))
    expect(getStorageBindingCount(source)).toBe(5);
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
    const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
    const sampler = kernels.get('reprojection-sample')!;
    expect(sampler).toContain('sub_fp64(projected.position.xy');
    expect(sampler).toContain('sub_fp64(projected.position.zw');
    for (const {source} of kernels.getConfigurations('reprojection-sample'))
      expect(getStorageBindingCount(source)).toBe(6);
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
        /^reprojection .*needs 204 bytes in one binding \(limit 203\)/
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
      /^reprojection projection parameters scratch needs .*limit/
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
    const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
    const sampler = kernels.get('reprojection-sample')!;
    expect(sampler).toContain('projected.valid!=0u');
    for (const {source} of kernels.getConfigurations('reprojection-sample'))
      expect(getStorageBindingCount(source)).toBe(6);
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
      /^reprojection affine parameters scratch needs 80 bytes in one binding \(limit 79\)/
    );
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    contributor.destroy();
    graph.device.destroy();
  }
});
