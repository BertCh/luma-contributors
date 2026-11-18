// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPURasterTileReprojection,
  type GPURasterTileReprojectionProps
} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {getGPURasterReprojectionParameterValues} from '../../src/raster-analysis/gpu-raster-reprojection';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

it.each([
  'source authority',
  'destination authority',
  'source wellKnownText',
  'destination wellKnownText',
  'source projectionJson',
  'destination projectionJson'
] as const)('final3 H3 tiled reprojection retains the captured %s', mutation => {
  const device = createNullWebGPUDevice();
  const run = (mutate: boolean) => {
    const graph = new GPUCommandGraph(device);
    const makeGrid = () => ({
      width: 1,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'point' as const,
      coordinateReferenceSystem: {
        authority: 'EPSG:4326',
        wellKnownText: 'GEOGCRS["WGS 84"]',
        projectionJson: {id: {authority: 'EPSG', code: 4326}, axis: [{name: 'latitude'}]}
      }
    });
    const sourceGrid = makeGrid();
    const destinationGrid = makeGrid();
    const contributor = new GPURasterTileReprojection({
      sourceGrid,
      destinationGrid,
      sources: [
        {
          pixelBounds: [0, 0, 1, 1],
          input: {
            id: 'one-cell',
            format: 'float32',
            storage: {kind: 'buffer', values: createTransientView(graph, 'one-cell', 'float32', 1)}
          }
        }
      ],
      output: {
        values: createTransientView(graph, 'values', 'float32', 1),
        validity: createTransientView(graph, 'validity', 'uint32', 1)
      }
    });
    try {
      if (mutate) {
        const crs = (mutation.startsWith('source') ? sourceGrid : destinationGrid)
          .coordinateReferenceSystem;
        if (mutation.endsWith('authority')) crs.authority = 'EPSG:3857';
        if (mutation.endsWith('wellKnownText'))
          crs.wellKnownText = 'PROJCRS["WGS 84 / Pseudo-Mercator"]';
        if (mutation.endsWith('projectionJson')) {
          crs.projectionJson.id.code = 3857;
          crs.projectionJson.axis[0].name = 'easting';
        }
      }
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(5);
      return getParsedKernelSources(nodes);
    } finally {
      contributor.destroy();
    }
  };
  try {
    const control = run(false);
    expect(run(true)).toEqual(control);
  } finally {
    device.destroy();
  }
});

it.each([
  'inherited grids',
  'getters'
] as const)('final2 G2 tiled reprojection captures %s once and routes the second cell', inherited => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const reads: number[] = [];
  const getterFields = <Fields extends object>(fields: Fields): Fields =>
    Object.create(
      null,
      Object.fromEntries(
        Object.entries(fields).map(([key, value]) => {
          const index = reads.push(0) - 1;
          return [
            key,
            {
              get: () => {
                reads[index]++;
                return reads[index] === 1 ? value : undefined;
              }
            }
          ];
        })
      )
    );
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  let props: GPURasterTileReprojectionProps = {
    sourceGrid: Object.create(grid),
    destinationGrid: Object.create(grid),
    sources: [
      {
        pixelBounds: [1, 0, 2, 1],
        input: {
          id: 'second-cell',
          format: 'float32',
          storage: {kind: 'buffer', values: createTransientView(graph, 'second-cell', 'float32', 1)}
        }
      }
    ],
    output: {
      values: createTransientView(graph, 'values', 'float32', 2),
      validity: createTransientView(graph, 'validity', 'uint32', 2)
    }
  };
  if (inherited === 'getters') {
    props = getterFields({
      ...props,
      sourceGrid: getterFields(grid),
      destinationGrid: getterFields(grid),
      sources: props.sources.map(tile => getterFields(tile)),
      output: getterFields(props.output)
    });
  }
  let contributor: GPURasterTileReprojection | undefined;
  try {
    contributor = new GPURasterTileReprojection(props);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(5);
    const route = getParsedKernelSources(nodes).get('raster-tile-reprojection-route-0')!;
    expect(route).toContain('let destinationIndex=(index/1u+0u)*2u+index%1u+1u');
    expect(reads.every(count => count === 1)).toBe(true);
  } finally {
    contributor?.destroy();
    device.destroy();
  }
});

it.each([
  'source dimensions',
  'destination dimensions',
  'source affine',
  'destination affine'
] as const)('final1 F4 tiled reprojection captures validated %s', mutation => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const sourceGrid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as [number, number, number, number, number, number],
    pixelInterpretation: 'point' as const
  };
  const destinationGrid = {
    ...sourceGrid,
    affine: [...sourceGrid.affine] as typeof sourceGrid.affine
  };
  const props = {
    sourceGrid,
    destinationGrid,
    sources: [
      {
        pixelBounds: [1, 0, 2, 1] as const,
        input: {
          id: 'second-cell',
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: createTransientView(graph, 'second-cell', 'float32', 1)
          }
        }
      }
    ],
    output: {
      values: createTransientView(graph, 'output-values', 'float32', 2),
      validity: createTransientView(graph, 'output-validity', 'uint32', 2)
    }
  };
  const originalParameters = getGPURasterReprojectionParameterValues(props);
  const contributor = new GPURasterTileReprojection(props);
  try {
    if (mutation === 'source dimensions') sourceGrid.width = 1;
    if (mutation === 'destination dimensions') destinationGrid.width = 1;
    if (mutation === 'source affine') sourceGrid.affine[2] = 1;
    if (mutation === 'destination affine') destinationGrid.affine[2] = 1;
    expect(contributor.props.sourceGrid.width).toBe(2);
    expect(contributor.props.destinationGrid.width).toBe(2);
    expect(contributor.props.sourceGrid.affine).toEqual([1, 0, 0, 0, 1, 0]);
    expect(contributor.props.destinationGrid.affine).toEqual([1, 0, 0, 0, 1, 0]);
    expect(getGPURasterReprojectionParameterValues(contributor.props)).toEqual(originalParameters);
    const nodes = contributor.getCommandNodes(graph);
    const clear = nodes.find(node => node.id === 'raster-tile-reprojection-clear')!;
    expect(
      clear.resources?.map(resource => ('buffer' in resource ? resource.buffer : undefined))
    ).toEqual([expect.objectContaining({length: 2}), expect.objectContaining({length: 2})]);
    const kernels = getParsedKernelSources(nodes);
    expect(kernels.get('raster-tile-reprojection-route-0')).toContain(
      'let destinationIndex=(index/1u+0u)*2u+index%1u+1u'
    );
  } finally {
    contributor.destroy();
    device.destroy();
  }
});

it.each([
  16382, 16383, 32768
])('enforces the sources.length command-node budget at %i tiles (RT-11)', sourceCount => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const tile = {
    pixelBounds: [0, 0, 1, 1] as const,
    input: {
      id: 'tile',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: createTransientView(graph, 'tile', 'float32', 1)}
    }
  };
  const output = {
    values: createTransientView(graph, 'values', 'float32', 1),
    validity: createTransientView(graph, 'validity', 'uint32', 1)
  };
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    const createContributor = () =>
      new GPURasterTileReprojection({
        id: 'tile-budget',
        sourceGrid: grid,
        destinationGrid: grid,
        sources: Array(sourceCount).fill(tile),
        output
      });
    if (sourceCount <= 16382) {
      const contributor = createContributor();
      expect(contributor.estimatedCommandNodeCount).toBe(32767);
      contributor.destroy();
    } else {
      expect(createContributor).toThrow(/^tile-budget .*sources.length.*32768/);
    }
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('tile routing parses many tiles with default-device storage binding limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const sourceGrid = {
    width: 32,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const sources = Array.from({length: 32}, (_, index) => ({
    pixelBounds: [index, 0, index + 1, 1] as const,
    input: {
      id: `tile-${index}`,
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: createTransientView(graph, `tile-${index}`, 'float32', 1)
      }
    }
  }));
  const props = {
    sourceGrid,
    destinationGrid: sourceGrid,
    sources,
    output: {
      values: createTransientView(graph, 'values', 'float32', 32),
      validity: createTransientView(graph, 'validity', 'uint32', 32)
    }
  };
  const contributor = new GPURasterTileReprojection(props);
  const nodes = contributor.getCommandNodes(graph);
  expect(contributor.estimatedCommandNodeCount).toBe(67);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const kernels = getParsedKernelSources(nodes);
  expect(kernels.size).toBeGreaterThan(32);
  for (const {id: name, source} of kernels.configurations)
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
  expect(
    () =>
      new GPURasterTileReprojection({
        ...props,
        sources: [{...sources[0], pixelBounds: [0.5, 0, 1, 1]}]
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterTileReprojection({
        ...props,
        sources: [{...sources[0], pixelBounds: [0, 0, 33, 1]}]
      })
  ).toThrow();
  contributor.destroy();
  graph.device.destroy();
});

it('rejects foreign parameters before mosaic allocation and permits a corrected retry', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const foreignGraph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const foreignParameters = createTransientView(foreignGraph, 'parameters', 'float32', 20);
  const localParameters = createTransientView(graph, 'parameters', 'float32', 20);
  const props = {
    id: 'tile-retry',
    sourceGrid: grid,
    destinationGrid: grid,
    sources: [],
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    }
  };
  const invalid = new GPURasterTileReprojection({...props, parameters: foreignParameters});
  const corrected = new GPURasterTileReprojection({...props, parameters: localParameters});
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => invalid.getCommandNodes(graph)).toThrow(/target graph/);
    const failedAllocations = allocations.mock.calls.length;
    expect(() => {
      for (const node of corrected.getCommandNodes(graph)) graph.add(node);
    }).not.toThrow();
    expect(failedAllocations).toBe(0);
  } finally {
    allocations.mockRestore();
    invalid.destroy();
    corrected.destroy();
    device.destroy();
  }
});

it('keeps the source list within the construction-time plan when the caller appends tiles', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const tile = {
    pixelBounds: [0, 0, 1, 1] as const,
    input: {
      id: 'tile',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: createTransientView(graph, 'tile', 'float32', 1)}
    }
  };
  const sources = [tile];
  const contributor = new GPURasterTileReprojection({
    sourceGrid: grid,
    destinationGrid: grid,
    sources,
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    }
  });
  sources.push(tile);
  try {
    expect(contributor.getCommandNodes(graph)).toHaveLength(5);
  } finally {
    contributor.destroy();
    device.destroy();
  }
});

it.each([
  'public list',
  'borrowed bounds',
  'borrowed input'
] as const)('routes the validated private snapshot after mutating %s (R3)', mutation => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const bounds: [number, number, number, number] = [0, 0, 1, 1];
  const original = createTransientView(graph, 'original', 'float32', 1);
  const tile = {
    pixelBounds: bounds,
    input: {
      id: 'tile',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: original}
    }
  };
  const contributor = new GPURasterTileReprojection({
    sourceGrid: grid,
    destinationGrid: grid,
    sources: [tile],
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    }
  });
  if (mutation === 'public list') contributor.props.sources = [tile, tile];
  if (mutation === 'borrowed bounds') {
    bounds[0] = 1;
    bounds[2] = 2;
  }
  if (mutation === 'borrowed input')
    tile.input.storage.values = createTransientView(graph, 'replacement', 'float32', 1);
  try {
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(5);
    const route = getParsedKernelSources(nodes).get('raster-tile-reprojection-route-0')!;
    expect(route).toContain('let destinationIndex=(index/1u+0u)*1u+index%1u+0u');
    const conversion = nodes.find(node => node.id === 'raster-tile-reprojection-tile-0-elevation')!;
    expect(
      conversion.resources?.some(resource => 'buffer' in resource && resource.buffer === original)
    ).toBe(true);
  } finally {
    contributor.destroy();
    device.destroy();
  }
});

it('rejects an oversized tile count before reading even the first tile (R5)', () => {
  const sources = new Array(32768);
  const readFirstTile = vi.fn(() => {
    throw new Error('tile getter must not run');
  });
  Object.defineProperty(sources, 0, {get: readFirstTile});
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  try {
    expect(
      () =>
        new GPURasterTileReprojection({
          id: 'budget-first',
          sourceGrid: grid,
          destinationGrid: grid,
          sources,
          output: {
            values: createTransientView(graph, 'values', 'float32', 1),
            validity: createTransientView(graph, 'validity', 'uint32', 1)
          }
        })
    ).toThrow(/^budget-first .*sources.length.*32768/);
    expect(readFirstTile).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
