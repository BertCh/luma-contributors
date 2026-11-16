// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {compactCells, getPentagons, polygonToCells, uncompactCells} from 'h3-js';
import {expect, it, vi} from 'vitest';
import {getS2ChildIndex, getS2IndexFromCell} from '@math.gl/dggs';
import {WgslReflect} from 'wgsl_reflect';
import {Buffer} from '@luma.gl/core';
import {createCompactionTestViewImporter} from './cell-compaction-test-views';
import {expectCellStorageScopes} from '../cell-indexing/cell-dispatch-audit';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  GPUCellCompaction,
  type GPUCellCompactionProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  bigIntToH3,
  h3ToBigInt,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {
  compactCellsOnCPU,
  compareBigInt,
  getCellDescendants,
  uncompactCellsOnCPU
} from './cell-compaction-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellCompactionProps> = {}
): GPUCellCompactionProps {
  const view = <Format extends 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    family: 'quadbin',
    operation: {type: 'compact', resolution: 6},
    cells: view('uint32x2', 32),
    output: {
      cells: view('uint32x2', 16),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    },
    ...overrides
  };
}

it('GPUCellCompaction validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const expectThrows = (overrides: Partial<GPUCellCompactionProps>, message: RegExp) =>
    expect(() => new GPUCellCompaction(createProps(graph, overrides))).toThrow(message);
  expectThrows({operation: {type: 'compact', resolution: 27}}, /resolution/);
  expectThrows({operation: {type: 'compact', resolution: 16}, family: 'h3'}, /resolution/);
  expectThrows(
    {operation: {type: 'compact', resolution: 6, minimumResolution: 7}},
    /minimumResolution/
  );
  expectThrows(
    {operation: {type: 'uncompact', resolution: 6, maximumDepth: 12}, family: 'h3'},
    /maximumDepth/
  );
  expectThrows({operation: {type: 'uncompact', resolution: 6, maximumDepth: -1}}, /maximumDepth/);
  expectThrows({cells: createTransientView(graph, 'bad-cells', 'uint32', 8) as never}, /cells/);
  expectThrows({mask: createTransientView(graph, 'bad-mask', 'uint32', 31)}, /mask/);
  expectThrows({wordOrder: 'big' as never}, /wordOrder/);
  const cells = createTransientView(graph, 'alias-cells', 'uint32x2', 8);
  expectThrows(
    {
      cells,
      output: {
        cells,
        status: {
          count: createTransientView(graph, 'alias-count', 'uint32', 1),
          overflow: createTransientView(graph, 'alias-overflow', 'uint32', 1)
        }
      }
    },
    /input|alias|disjoint|overlap/i
  );
  expectThrows(
    {
      output: {
        ...createProps(graph).output,
        status: {
          ...createProps(graph).output.status,
          count: createTransientView(graph, 'empty-count', 'uint32', 0)
        }
      }
    },
    /one uint32 row/
  );
  device.destroy();
});

it('GPUCellCompaction emits deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const getIds = (overrides: Partial<GPUCellCompactionProps>) => {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUCellCompaction(createProps(graph, {id: 'compaction', ...overrides}));
    return contributor.getCommandNodes(graph).map(node => node.id);
  };
  const compact = getIds({});
  expect(getIds({})).toEqual(compact);
  expect(compact[0]).toBe('compaction-keys');
  expect(compact.at(-1)).toBe('compaction-emit');
  expect(compact).toContain('compaction-analyze');
  expect(compact).toContain('compaction-scatter');
  const sortedIds = getIds({operation: {type: 'compact', resolution: 6, sorted: true}});
  expect(sortedIds.some(id => id.startsWith('compaction-table-sort'))).toBe(false);
  expect(compact.some(id => id.startsWith('compaction-table-sort'))).toBe(true);
  const uncompact = getIds({operation: {type: 'uncompact', resolution: 8}});
  expect(uncompact[0]).toBe('compaction-count');
  expect(uncompact.at(-1)).toBe('compaction-expand');
  expect(getIds({operation: {type: 'uncompact', resolution: 8}})).toEqual(uncompact);
  device.destroy();
});

it('the compaction oracle matches h3-js compactCells and uncompactCells', () => {
  const polygon = [
    [37.5, -122.8],
    [38.1, -122.8],
    [38.1, -122.0],
    [37.5, -122.0]
  ];
  for (const resolution of [5, 6, 7]) {
    const cells = polygonToCells(polygon, resolution);
    const compacted = compactCells(cells).map(h3ToBigInt).sort(compareBigInt);
    expect(compactCellsOnCPU('h3', cells.map(h3ToBigInt), resolution)).toEqual(compacted);
    expect(compacted.length).toBeLessThan(cells.length);
    const expanded = uncompactCells(compactCells(cells), resolution).map(h3ToBigInt);
    expect(uncompactCellsOnCPU('h3', compacted, resolution).sort(compareBigInt)).toEqual(
      expanded.sort(compareBigInt)
    );
  }
  // Pentagon: only its 6 children (no K child) compact.
  const [pentagon] = getPentagons(4);
  const children = getCellDescendants('h3', h3ToBigInt(pentagon), 5);
  expect(children).toHaveLength(6);
  expect(compactCellsOnCPU('h3', children, 5)).toEqual([h3ToBigInt(pentagon)]);
  expect(bigIntToH3(compactCellsOnCPU('h3', children, 5, 5)[0])).toBe(bigIntToH3(children[0]));
});

it('the Quadbin oracle compacts complete blocks and keeps partial ones', () => {
  const block: bigint[] = [];
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) {
      block.push(quadbinTileToCell(4 + x, 8 + y, 5));
    }
  }
  expect(compactCellsOnCPU('quadbin', block, 5)).toEqual([quadbinTileToCell(1, 2, 3)]);
  expect(compactCellsOnCPU('quadbin', block, 5, 4)).toEqual(
    getCellDescendants('quadbin', quadbinTileToCell(1, 2, 3), 4)
  );
  const partial = block.slice(1);
  // Dropping one tile keeps its 3 siblings, the other 3 sibling groups become res 4 parents.
  const compactedPartial = compactCellsOnCPU('quadbin', partial, 5);
  expect(compactedPartial).toHaveLength(6);
  expect(compactCellsOnCPU('quadbin', [...block, ...block], 5)).toEqual([
    quadbinTileToCell(1, 2, 3)
  ]);
});

it('S2 compaction oracle recursively merges independently constructed Hilbert children through level 30', () => {
  for (const resolution of [2, 26, 30]) {
    for (let face = 0; face < 6; face++) {
      const parent = getS2IndexFromCell({
        face,
        ij: [13 % 2 ** (resolution - 2), 7 % 2 ** (resolution - 2)],
        level: resolution - 2
      });
      const children = [0, 1, 2, 3].map(slot => getS2ChildIndex(parent, slot));
      const leaves = children.flatMap(cell =>
        [0, 1, 2, 3].map(slot => getS2ChildIndex(cell, slot))
      );
      expect(getCellDescendants('s2', parent, resolution)).toEqual(leaves);
      expect(compactCellsOnCPU('s2', [...leaves, ...leaves], resolution)).toEqual([parent]);
      expect(compactCellsOnCPU('s2', leaves, resolution, resolution - 1)).toEqual(children);
      expect(compactCellsOnCPU('s2', leaves.slice(1), resolution)).toEqual([
        ...leaves.slice(1, 4),
        ...children.slice(1)
      ]);
      expect(uncompactCellsOnCPU('s2', [parent], resolution)).toEqual(leaves);
    }
  }
});

it('GPUCellCompaction rejects overlapping aligned writable ranges, including distinct views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'output-backing', 'uint32', 256);
    const view = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset});
    for (const byteOffset of [0, 4, 252]) {
      expect(
        () =>
          new GPUCellCompaction({
            ...props,
            output: {...props.output, status: {count: view(0), overflow: view(byteOffset)}}
          })
      ).toThrow(/overlap/);
    }
    expect(
      () =>
        new GPUCellCompaction({
          ...props,
          output: {...props.output, status: {count: view(0), overflow: view(256)}}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('GPUCellCompaction preflights aligned inputs, outputs and parameter scratch before declaring resources', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 256});
    for (const field of ['cells', 'outputCells', 'inputCount', 'parameters'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(
        graph,
        field === 'parameters'
          ? {cells: createTransientView(graph, 'single-cell', 'uint32x2', 1)}
          : {}
      );
      if (field === 'parameters')
        props.output.cells = createTransientView(graph, 'single-output', 'uint32x2', 1);
      const backing = createTransientView(graph, 'binding-backing', 'uint32', 256);
      const oversized = graph.createDataView(backing.buffer, {
        format: 'uint32x2',
        length: 32,
        byteOffset: 4
      });
      const contributor = new GPUCellCompaction({
        ...props,
        cells: field === 'cells' ? oversized : props.cells,
        count:
          field === 'inputCount'
            ? graph.createDataView(backing.buffer, {format: 'uint32', length: 65})
            : undefined,
        output: field === 'outputCells' ? {...props.output, cells: oversized} : props.output
      });
      if (field === 'parameters')
        Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
          value: 64,
          configurable: true
        });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      const add = vi.spyOn(graph, 'add');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        field === 'parameters' ? /cell table parameters.*limit 64/ : /limit 256/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
      vi.restoreAllMocks();
    }
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('GPUCellCompaction emits a monotone saturated uncompact scan instead of total-only carry detection', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUCellCompaction(
      createProps(graph, {operation: {type: 'uncompact', resolution: 15, maximumDepth: 15}})
    ).getCommandNodes(graph);
    expect(nodes.map(node => node.id)).toContain('cell-compaction-scan-saturate');
    expect(nodes.some(node => node.workload?.operation === 'GPUScanUint64')).toBe(true);
  } finally {
    device.destroy();
  }
});

it('GPUCellCompaction validates numeric and family props with the contributor ID', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (const resolution of [NaN, Infinity, -1, 30.5, 31]) {
      expect(
        () =>
          new GPUCellCompaction(
            createProps(graph, {
              id: 'checked',
              family: 's2',
              operation: {type: 'compact', resolution}
            })
          )
      ).toThrow(/^checked .*resolution/);
    }
    expect(
      () => new GPUCellCompaction(createProps(graph, {id: 'checked', family: 'bogus' as never}))
    ).toThrow(/^checked family/);
    expect(
      () =>
        new GPUCellCompaction(
          createProps(graph, {
            id: 'checked',
            family: 's2',
            operation: {type: 'uncompact', resolution: 30, maximumDepth: 16}
          })
        )
    ).toThrow(/^checked maximumDepth/);
  } finally {
    device.destroy();
  }
});

it('GPUCellCompaction S2 shaders parse and contain no W3C reserved identifiers', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const type of ['compact', 'uncompact'] as const) {
      const graph = new GPUCommandGraph(device);
      const nodes = new GPUCellCompaction(
        createProps(graph, {family: 's2', operation: {type, resolution: 30}})
      ).getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(kernels.some(kernel => kernel.operation === 'GPUCellCompaction')).toBe(true);
      for (const {source} of kernels) {
        expect(() => new WgslReflect(source)).not.toThrow();
        const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
        expect(
          (withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
            WGSL_RESERVED_WORDS.has(identifier)
          )
        ).toEqual([]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('GPUCellCompaction S2 uses a fixed pipeline count as the number of hierarchy levels grows', () => {
  const device = createNullWebGPUDevice();
  try {
    const counts: number[] = [];
    for (const resolution of [0, 3, 26, 30]) {
      const graph = new GPUCommandGraph(device);
      const nodes = new GPUCellCompaction(
        createProps(graph, {family: 's2', operation: {type: 'compact', resolution}})
      ).getCommandNodes(graph);
      counts.push(getGPUCommandNodePipelineKeys(nodes).size);
      expect(nodes.filter(node => node.id.endsWith('-analyze'))).toHaveLength(1);
    }
    expect(new Set(counts).size).toBe(1);
  } finally {
    device.destroy();
  }
});

it('allows disjoint packed status bytes used in separate compaction dispatches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'packed-status', 'uint32', 129);
    const view = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset});
    const output = {
      ...props.output,
      status: {count: view(4), droppedCount: view(8), overflow: view(260), requiredCount: view(512)}
    };
    const contributor = new GPUCellCompaction({...props, output});
    expectCellStorageScopes(contributor.getCommandNodes(graph));
    expect(
      () =>
        new GPUCellCompaction({
          ...props,
          output: {...output, status: {...output.status, droppedCount: view(4)}}
        })
    ).toThrow(/logical.*overlap/);
  } finally {
    device.destroy();
  }
});

it('the nonaligned compaction fixture imports its physical output buffer exactly once', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE | Buffer.COPY_SRC});
  try {
    const graph = new GPUCommandGraph(device);
    const imported = vi.spyOn(graph, 'importBuffer');
    const track = vi.fn((value: Buffer) => value);
    const importView = createCompactionTestViewImporter(
      graph,
      track,
      {cells: 4, count: 260, dropped: 264, overflow: 516, total: 772},
      true
    );
    const cells = importView('cells', buffer, 'uint32x2', 4);
    const count = importView('count', buffer, 'uint32', 1);
    const droppedCount = importView('dropped', buffer, 'uint32', 1);
    const overflow = importView('overflow', buffer, 'uint32', 1);
    const requiredCount = importView('total', buffer, 'uint32', 1);
    expect(imported).toHaveBeenCalledTimes(1);
    expect(track).toHaveBeenCalledTimes(1);
    for (const view of [count, droppedCount, overflow, requiredCount])
      expect(view.buffer).toBe(cells.buffer);
    expect(cells.byteOffset).toBe(4);
    expect(count.byteOffset).toBe(260);
    const contributor = new GPUCellCompaction({
      ...createProps(graph),
      family: 's2',
      output: {cells, status: {count, droppedCount, overflow, requiredCount}}
    });
    expectCellStorageScopes(contributor.getCommandNodes(graph));
  } finally {
    vi.restoreAllMocks();
    buffer.destroy();
    device.destroy();
  }
});
