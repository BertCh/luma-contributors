// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUTerrainTiledFlow,
  getTerrainTiledFlowExecutionPlan,
  type GPUTerrainTiledFlowProps,
  type GPUTerrainFlowTile
} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {
  planTerrainD8Tiles,
  TERRAIN_TILED_MAXIMUM_CELL_COUNT,
  TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT
} from '../../../src/gpu-terrain/terrain-processing/terrain-tiled-plan';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function makeTiles(
  graph: GPUCommandGraph,
  widths: number[],
  height = 1,
  branching = false
): GPUTerrainFlowTile[] {
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `tile-field-${identifier++}`, format, length);
  const count = widths.reduce((total, width) => total + width, 0) * height;
  // Consecutive chunks share a binding-alignment window, but are written in different dispatches.
  const accumulation = view('float32', count);
  const status = view('uint32', count);
  const reachIds = view('uint32', count);
  let column = 0;
  return widths.map((width, tileIndex) => {
    const length = width * height;
    const offset = column * height * 4;
    const slice = <Format extends 'float32' | 'uint32'>(backing: GraphDataView<Format>) =>
      graph.createDataView(backing.buffer, {format: backing.format, byteOffset: offset, length});
    // Deliberately different binding prefixes, including non-256-byte-aligned starts.
    const prefixed = <Format extends 'float32' | 'uint32'>(format: Format, rowCount = length) => {
      const backing = view(format, rowCount + tileIndex + 1);
      return graph.createDataView(backing.buffer, {
        format,
        byteOffset: (tileIndex + 1) * 4,
        length: rowCount
      });
    };
    const tile: GPUTerrainFlowTile = {
      column,
      row: 0,
      width,
      height,
      owned: [0, 0, width, height],
      flowDirections: prefixed('uint32'),
      runoff: prefixed('float32'),
      streams: prefixed('uint32'),
      ...(branching ? {fractions: prefixed('float32', length * 8)} : {}),
      output: {accumulation: slice(accumulation), status: slice(status), reachIds: slice(reachIds)}
    };
    column += width;
    return tile;
  });
}

function makeGrid(width: number, height = 1) {
  return {width, height, affine: [1, 0, 0, 0, 1, 0] as const, pixelInterpretation: 'area' as const};
}

it('review B4 captures inherited terrain route, tile and output fields', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const tile = makeTiles(graph, [1])[0];
    const inheritedTile = Object.create(tile, {output: {get: () => Object.create(tile.output)}});
    const props: GPUTerrainTiledFlowProps = Object.create(
      {grid: makeGrid(1), tiles: [inheritedTile]},
      {
        execution: {get: () => 'tiled'},
        maximumIterations: {get: () => 1}
      }
    );
    expect(getTerrainTiledFlowExecutionPlan(props, device).selectedExecution).toBe('tiled');
    const contributor = new GPUTerrainTiledFlow(props);
    expect(contributor.selectedExecution).toBe('tiled');
    expect(contributor.props.maximumIterations).toBe(1);
    expect(contributor.props.tiles[0].output.status).toBe(tile.output.status);
  } finally {
    device.destroy();
  }
});

it('review B6 documents graph-node budgets separately from GPU commands', () => {
  const page = readFileSync(
    new URL('../../../../../docs/api/gpu-terrain/terrain-processing.md', import.meta.url),
    'utf8'
  );
  expect(page).toContain('**32,768 command nodes**');
  expect(page).toContain('3,072-command-node estimate');
  const source = readFileSync(
    new URL(
      '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(source).toContain('command-node estimate must be at most 32,768');
});

it('SCL-07 budgets only the resident route selected for one cell', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUTerrainTiledFlow({
      grid: makeGrid(1),
      tiles: makeTiles(graph, [1]),
      maximumIterations: 1
    });
    expect(contributor.selectedExecution).toBe('resident');
    expect(contributor.estimatedCommandNodeCount).toBe(17); // 8 solver + 2 exchange/publication + 1 reach + 0 jumps + 6 status allowance
    expect(contributor.getCommandNodes(graph)).toHaveLength(11);
    expect(contributor.worstCaseCommandNodeCount).toBe(517);
  } finally {
    device.destroy();
  }
});

it('SCL-07 admits 6452 one-cell tiles when the resident fraction plane fits', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUTerrainTiledFlow({
      grid: makeGrid(6452),
      tiles: makeTiles(graph, Array(6452).fill(1)),
      maximumIterations: 1
    });
    expect(contributor.selectedExecution).toBe('resident');
    expect(contributor.estimatedCommandNodeCount).toBe(12932); // 8 + 2*6452 + 1 + ceil(log2(6452)) + 6
    expect(contributor.worstCaseCommandNodeCount).toBe(32772); // 5*6452 + 512, unused tiled route
  } finally {
    device.destroy();
  }
});

it('SCL-15 keeps 32768 by 16384 on the tiled side of the checked cell bounds', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const tile = makeTiles(graph, [1024], 1024)[0];
    const props = {
      grid: makeGrid(32768, 16384),
      execution: 'tiled' as const,
      tiles: Array.from({length: 512}, (_, index) => ({
        ...tile,
        column: (index % 32) * 1024,
        row: Math.floor(index / 32) * 1024,
        output: {
          accumulation: createTransientView(
            graph,
            `large-accumulation-${index}`,
            'float32',
            1048576
          ),
          status: createTransientView(graph, `large-status-${index}`, 'uint32', 1048576),
          reachIds: createTransientView(graph, `large-reaches-${index}`, 'uint32', 1048576)
        }
      })),
      maximumIterations: 1
    };
    // 32*16 cores, each 1024^2 cells. Each square has 4*1024-4 boundary cells.
    expect(TERRAIN_TILED_MAXIMUM_CELL_COUNT).toBe(536870911);
    expect(TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT).toBe(4294967295);
    const plan = planTerrainD8Tiles('documented-grid', props);
    expect(plan.maximumCellCount).toBe(1048576);
    expect(plan.boundaryCount).toBe(2095104);
    expect(plan.estimatedCommandNodeCount).toBe(3072);
    expect(new GPUTerrainTiledFlow(props).estimatedCommandNodeCount).toBe(3072);
    expect(() => new GPUTerrainTiledFlow({...props, execution: 'resident'})).toThrow(
      /resident grid cell count.*536870911/
    );
  } finally {
    device.destroy();
  }
});

for (const branching of [false, true]) {
  it(`keeps one exchange and one publication pipeline for unequal tile counts (branching=${branching})`, () => {
    const device = createNullWebGPUDevice();
    try {
      for (const widths of [[260], [1, 2, 257]]) {
        const graph = new GPUCommandGraph(device);
        const contributor = new GPUTerrainTiledFlow({
          grid: makeGrid(260),
          tiles: makeTiles(graph, widths, 1, branching),
          routing: branching ? 'branching' : 'd8',
          maximumIterations: 1
        });
        const nodes = contributor.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        const exchange = nodes.filter(node => /-exchange-owned-\d+$/.test(node.id));
        const publication = nodes.filter(node => /^terrain-tiled-flow-publish-\d+$/.test(node.id));
        for (const family of [exchange, publication]) {
          expect(family).toHaveLength(widths.length);
          expect(getGPUCommandNodePipelineKeys(family, {includeShaderLayout: true}).size).toBe(1);
          expect(family.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
            widths.map(width => Math.ceil(width / 256))
          );
          for (const {id: identifier, source, constants} of getParsedKernelSources(family)
            .configurations) {
            expect(auditWGSLModule(source, constants), identifier).toEqual([]);
            expect(source).not.toContain('override');
          }
        }
      }
    } finally {
      device.destroy();
    }
  });
}

it('rejects aligned output overlap within a tile and logical overlap across tiles', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const sameTile of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const tiles = makeTiles(graph, [1, 2]);
      const backing = createTransientView(graph, 'overlap', 'uint32', 4);
      tiles[0].output.accumulation = graph.createDataView(backing.buffer, {
        format: 'float32',
        length: 1
      });
      tiles[sameTile ? 0 : 1].output.status = graph.createDataView(backing.buffer, {
        format: 'uint32',
        byteOffset: sameTile ? 4 : 0,
        length: sameTile ? 1 : 2
      });
      expect(
        () => new GPUTerrainTiledFlow({grid: makeGrid(3), tiles, maximumIterations: 1})
      ).toThrow(sameTile ? /aligned binding/ : /logical/);
    }
  } finally {
    device.destroy();
  }
});

it('rejects separate imported output handles that alias the same physical buffer', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE});
  try {
    const graph = new GPUCommandGraph(device);
    const tiles = makeTiles(graph, [2]);
    tiles[0].output.accumulation = importGraphBuffer(
      graph,
      'aliased-accumulation',
      buffer,
      'float32',
      2
    );
    tiles[0].output.status = importGraphBuffer(graph, 'aliased-status', buffer, 'uint32', 2);
    expect(() => new GPUTerrainTiledFlow({grid: makeGrid(2), tiles, maximumIterations: 1})).toThrow(
      /overlap/
    );
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('preflights resident fractions before declaring graph resources above the core binding ceiling', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    // 2049 * 2048 cells exceed the 128 MiB, eight-fraction resident plane.
    const contributor = new GPUTerrainTiledFlow({
      grid: makeGrid(2049, 2048),
      execution: 'resident',
      tiles: makeTiles(graph, [1024, 1025], 2048),
      maximumIterations: 1
    });
    const declarations = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/resident fractions scratch/);
    expect(declarations).not.toHaveBeenCalled();
    declarations.mockRestore();
  } finally {
    device.destroy();
  }
});

it('checks the aligned binding prefix before any resident scratch is declared', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const length = device.limits.maxStorageBufferBindingSize / 4;
    const backing = createTransientView(graph, 'oversized-input', 'float32', length + 1);
    // Logical bytes fit exactly, but the four-byte binding prefix exceeds the limit.
    const fractions = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 4,
      length
    });
    const contributor = new GPUTerrainBranchingFlow({
      grid: makeGrid(length / 8),
      fractions,
      values: createTransientView(graph, 'values', 'float32', length / 8),
      maximumIterations: 1,
      mode: 'transport',
      output: {
        values: createTransientView(graph, 'output', 'float32', length / 8),
        status: createTransientView(graph, 'status', 'uint32', length / 8),
        unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
      }
    });
    const declarations = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/aligned|binding/);
    expect(declarations).not.toHaveBeenCalled();
    declarations.mockRestore();
  } finally {
    device.destroy();
  }
});

for (const iterations of [1, 4, 9]) {
  it(`bounds branching command nodes including validity and ${iterations} sweeps`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const tile = makeTiles(graph, [3], 1, true)[0];
      const contributor = new GPUTerrainBranchingFlow({
        grid: makeGrid(3),
        fractions: tile.fractions!,
        values: tile.runoff,
        validity: tile.streams,
        maximumIterations: iterations,
        mode: 'transport',
        output: {
          values: tile.output.accumulation,
          status: tile.output.status,
          unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
        }
      });
      expect(contributor.getCommandNodes(graph)).toHaveLength(
        contributor.estimatedCommandNodeCount
      );
    } finally {
      device.destroy();
    }
  });
}

it('uniform terrain measures use integer coordinate differences and ignore the unused world origin', async () => {
  const {getTerrainGroundMeasureWGSL} = await import(
    '../../../src/gpu-terrain/terrain-processing/terrain-processing-utils'
  );
  const source = getTerrainGroundMeasureWGSL({
    ...makeGrid(16777218, 2),
    affine: [3, 4, Number.MAX_VALUE, 0, 2, -Number.MAX_VALUE]
  });
  expect(source).toContain('f32(first - second)');
  expect(source).not.toContain('f32(other % GRID_WIDTH) - f32(cell % GRID_WIDTH)');
  expect(source).not.toContain(String(Number.MAX_VALUE));
  expect(
    auditWGSLModule(`const GRID_WIDTH: u32 = 16777218u;\nconst GRID_HEIGHT: u32 = 2u;\n${source}`)
  ).toEqual([]);
});

for (const mode of ['transport', 'fractional-catchment'] as const) {
  it(`does not compile unused affine distance arithmetic for ${mode}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const tile = makeTiles(graph, [3], 1, true)[0];
      const contributor = new GPUTerrainBranchingFlow({
        grid: {...makeGrid(3), affine: [Number.MAX_VALUE, 0, 0, 0, Number.MAX_VALUE, 0]},
        fractions: tile.fractions!,
        values: tile.runoff,
        maximumIterations: 1,
        mode,
        output: {
          values: tile.output.accumulation,
          status: tile.output.status,
          unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
        }
      });
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      for (const {id: identifier, source, constants} of sources.configurations) {
        expect(source).not.toContain('fn getGroundEdgeLength');
        expect(auditWGSLModule(source, constants), identifier).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  });
}

it('checks the tiled command budget before visiting a large tile list', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const tile = makeTiles(graph, [1])[0];
    expect(
      () =>
        new GPUTerrainTiledFlow({
          grid: makeGrid(1),
          tiles: Array(16384).fill(tile),
          maximumIterations: 1
        })
    ).toThrow(/estimated command node count.*32768/);
  } finally {
    device.destroy();
  }
});

it('shared terrain validation rejects physical output aliases in branching flow', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 12, usage: Buffer.STORAGE});
  try {
    const graph = new GPUCommandGraph(device);
    const tile = makeTiles(graph, [3], 1, true)[0];
    expect(
      () =>
        new GPUTerrainBranchingFlow({
          grid: makeGrid(3),
          fractions: tile.fractions!,
          values: tile.runoff,
          maximumIterations: 1,
          mode: 'transport',
          output: {
            values: importGraphBuffer(graph, 'values-alias', buffer, 'float32', 3),
            status: importGraphBuffer(graph, 'status-alias', buffer, 'uint32', 3),
            unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
          }
        })
    ).toThrow(/overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

for (const field of ['elevationDecrement', 'maximumCutDepth'] as const)
  it(`retains construction-time rejection of nonrepresentable breach ${field}`, async () => {
    const {GPUTerrainBreach} = await import(
      '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach'
    );
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      expect(
        () =>
          new GPUTerrainBreach({
            grid: makeGrid(1),
            elevation: view('source', 'float32', 1),
            pitCells: view('pits', 'uint32', 1),
            maximumSteps: 1,
            maximumCutDepth: field === 'maximumCutDepth' ? Number.MAX_VALUE : 1,
            elevationDecrement: field === 'elevationDecrement' ? Number.MAX_VALUE : 0.25,
            output: {
              elevation: view('edited', 'float32', 1),
              cutDepth: view('depth', 'float32', 1),
              modificationMask: view('mask', 'uint32', 1),
              pathCells: view('paths', 'uint32', 2),
              pathLengths: view('lengths', 'uint32', 1),
              pitStatus: view('status', 'uint32', 1),
              unresolvedCount: view('unresolved', 'uint32', 1)
            }
          })
      ).toThrow(new RegExp(`terrain-breach ${field} must`));
    } finally {
      device.destroy();
    }
  });

it('does not compile unused path kernels for a burn-only breach with the largest step bound', async () => {
  const {GPUTerrainBreach} = await import(
    '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const contributor = new GPUTerrainBreach({
      grid: makeGrid(1),
      elevation: view('source', 'float32', 1),
      pitCells: view('pits', 'uint32', 0),
      maximumSteps: 0xffffffff,
      maximumCutDepth: 1,
      elevationDecrement: 0.25,
      burnDepth: view('burn', 'float32', 1),
      output: {
        elevation: view('edited', 'float32', 1),
        cutDepth: view('depth', 'float32', 1),
        modificationMask: view('mask', 'uint32', 1),
        pathCells: view('paths', 'uint32', 0),
        pathLengths: view('lengths', 'uint32', 0),
        pitStatus: view('status', 'uint32', 0),
        unresolvedCount: view('unresolved', 'uint32', 1)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.some(node => /edit-proposals|verify-paths/.test(node.id))).toBe(false);
    for (const {id: identifier, source, constants} of getParsedKernelSources(nodes)
      .configurations) {
      expect(source).not.toContain('4294967296u');
      expect(auditWGSLModule(source, constants), identifier).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
