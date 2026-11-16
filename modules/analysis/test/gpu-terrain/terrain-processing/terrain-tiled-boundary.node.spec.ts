// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUTerrainTiledFlow,
  type GPUTerrainFlowTile
} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {
  planTerrainD8Tiles,
  validateTerrainD8TileBindings
} from '../../../src/gpu-terrain/terrain-processing/terrain-tiled-plan';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createTerrainTileSequence} from '../../../src/gpu-terrain/terrain-processing/terrain-tiled-sequence';

it('declares the union of packed caller input ranges in a tile scheduling node', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const backing = createTransientView(graph, 'packed-input', 'uint32', 80);
    const first = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 3
    });
    const second = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 260,
      length: 4
    });
    const sequence = createTerrainTileSequence('tile', [
      {
        repetitions: 1,
        nodes: [
          createGPUComputeCommandNode({
            id: 'read-packed-inputs',
            resources: [
              {buffer: first, usage: 'storage-read'},
              {buffer: second, usage: 'storage-read'}
            ],
            compile: () => ({encode() {}})
          })
        ]
      }
    ]);
    expect(sequence.resources).toHaveLength(1);
    expect(sequence.resources?.[0]).toMatchObject({
      buffer: {buffer: backing.buffer, byteOffset: 4, length: 68},
      usage: 'storage-read'
    });
  } finally {
    device.destroy();
  }
});

function setup(
  widths: number[],
  height: number,
  execution: 'auto' | 'resident' | 'tiled' = 'tiled'
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, count: number) =>
    createTransientView(graph, `field-${identifier++}`, format, count);
  let column = 0;
  const tiles: GPUTerrainFlowTile[] = widths.map((width, tileIndex) => {
    const count = width * height;
    const prefixed = <Format extends 'float32' | 'uint32'>(format: Format) => {
      const backing = view(format, count + tileIndex + 1);
      return graph.createDataView(backing.buffer, {
        format,
        length: count,
        byteOffset: (tileIndex + 1) * 4
      });
    };
    const tile = {
      column,
      row: 0,
      width,
      height,
      owned: [0, 0, width, height] as const,
      flowDirections: prefixed('uint32'),
      streams: prefixed('uint32'),
      runoff: prefixed('float32'),
      output: {
        accumulation: prefixed('float32'),
        status: prefixed('uint32'),
        reachIds: prefixed('uint32')
      }
    };
    column += width;
    return tile;
  });
  const props = {
    grid: {
      width: column,
      height,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    },
    execution,
    tiles,
    maximumIterations: 1
  };
  return {device, graph, props};
}

it('bounds tile scratch independently of the global cell count above the old resident ceiling', () => {
  const {device, graph, props} = setup([1024, 1025], 2048, 'auto');
  try {
    const contributor = new GPUTerrainTiledFlow(props);
    const allocations = vi.spyOn(graph, 'createTransientBuffer');
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.some(node => /boundary-solve$/.test(node.id))).toBe(true);
    expect(nodes.some(node => /exchange-owned-/.test(node.id))).toBe(false);
    const plan = planTerrainD8Tiles('test', props);
    expect(plan.boundaryCount).toBe(12282);
    expect(plan.maximumCellCount).toBe(1025 * 2048);
    const sizes = allocations.mock.calls.map(([descriptor]) => descriptor.byteLength);
    expect(Math.max(...sizes)).toBe(plan.maximumCellCount * 32);
    expect(
      allocations.mock.calls.some(
        ([descriptor]) => descriptor.id === 'terrain-tiled-flow-fractions'
      )
    ).toBe(false);
    allocations.mockRestore();
  } finally {
    device.destroy();
  }
});

function terrainPipelines(nodes: readonly GPUCommandNode[]) {
  return new Set(
    getGPUCommandNodeKernelSources(nodes)
      .filter(kernel => kernel.operation === 'GPUTerrainTiledFlow')
      .map(
        kernel =>
          `${kernel.source}\n${JSON.stringify(kernel.constants)}\n${JSON.stringify(kernel.shaderLayout)}`
      )
  );
}

for (const height of [1, 3])
  it(`uses constant terrain shaders and pipelines across uneven tiles at height ${height}`, () => {
    const sets: Set<string>[] = [];
    for (const widths of [[768], [255, 256, 257]]) {
      const {device, graph, props} = setup(widths, height);
      try {
        const contributor = new GPUTerrainTiledFlow(props);
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        sets.push(terrainPipelines(nodes));
        const publications = nodes.filter(node => /publish-bounded-\d+$/.test(node.id));
        expect(publications.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
          widths.map(width => Math.ceil((width * height) / 256))
        );
        for (const [name, source] of getParsedKernelSources(nodes)) {
          expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
          expect(auditWGSLModule(source), name).toEqual([]);
        }
      } finally {
        device.destroy();
      }
    }
    expect(sets[1].size).toBe(sets[0].size);
    expect([...sets[1]].sort()).toEqual([...sets[0]].sort());
  });

it('prechecks the coarse boundary forest before declaring tile resources', () => {
  const {device, graph, props} = setup([4194305], 1);
  try {
    const contributor = new GPUTerrainTiledFlow(props);
    const allocations = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/tour sums scratch/);
    expect(allocations).not.toHaveBeenCalled();
    allocations.mockRestore();
  } finally {
    device.destroy();
  }
});

it('plans bounded scratch for a 30000 x 30000 D8 domain', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const input = createTransientView(graph, 'directions', 'uint32', 1000000);
    const runoff = createTransientView(graph, 'runoff', 'float32', 1000000);
    const tiles: GPUTerrainFlowTile[] = [];
    for (let row = 0; row < 30; row++)
      for (let column = 0; column < 30; column++) {
        const prefix = `tile-${row}-${column}`;
        tiles.push({
          column: column * 1000,
          row: row * 1000,
          width: 1000,
          height: 1000,
          owned: [0, 0, 1000, 1000],
          flowDirections: input,
          streams: input,
          runoff,
          output: {
            accumulation: createTransientView(graph, `${prefix}-accumulation`, 'float32', 1000000),
            status: createTransientView(graph, `${prefix}-status`, 'uint32', 1000000),
            reachIds: createTransientView(graph, `${prefix}-reach`, 'uint32', 1000000)
          }
        });
      }
    const props = {
      grid: {
        width: 30000,
        height: 30000,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      tiles,
      execution: 'tiled' as const
    };
    const contributor = new GPUTerrainTiledFlow(props);
    const plan = planTerrainD8Tiles(contributor.id, props);
    expect(plan.maximumCellCount).toBe(1000000);
    expect(plan.boundaryCount).toBe(3596400);
    expect(plan.boundaryCount * 32).toBeLessThanOrEqual(134217728);
    expect(contributor.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
    expect(() => validateTerrainD8TileBindings(graph, contributor.id, plan)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('compiles the tile schedule with graph hazard and resource validation on a recording device', () => {
  const {device, graph, props} = setup([3, 4], 3);
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const pipeline = vi
    .spyOn(device, 'createComputePipeline')
    .mockImplementation(
      properties =>
        ({id: properties.id ?? 'recorded', hash: '', destroy() {}}) as ReturnType<
          typeof device.createComputePipeline
        >
    );
  try {
    graph.add(new GPUTerrainTiledFlow(props));
    const compiled = graph.compile();
    expect(pipeline).toHaveBeenCalled();
    compiled.destroy();
  } finally {
    pipeline.mockRestore();
    device.destroy();
  }
});
