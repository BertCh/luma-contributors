// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import type {GPUCellTable} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-table';
import {
  GPUCellTableCompare,
  type GPUCellTableCompareProps
} from '../../../src/gpu-spatial-analysis/cell-table-compare/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  compareCellTablesOnCPU,
  convertInt64ToFloat32,
  sumFixedOrder,
  type CompareInputCell
} from './cell-table-compare-oracle';
import {
  expectCellBarrierEntry,
  expectCellStorageScopes
} from '../cell-indexing/cell-dispatch-audit';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

let serial = 0;

function createTable(graph: GPUCommandGraph, capacity: number, withSums = true): GPUCellTable {
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    cells: view('uint32x2', capacity),
    counts: view('uint32', capacity),
    sums: withSums ? view('uint32x2', capacity) : undefined,
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellTableCompareProps> = {}
): GPUCellTableCompareProps {
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    before: createTable(graph, 4),
    after: createTable(graph, 6),
    output: {
      cells: view('uint32x2', 8),
      presence: view('uint32', 8),
      delta: view('float32', 8),
      zScore: view('float32', 8),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUCellTableCompareProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUCellTableCompare(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const cell = (key: bigint, count: number, sum = 0n): CompareInputCell => ({
  key,
  count,
  sum
});
const options = {
  measure: 'count',
  sumScale: 65536,
  zScore: 'poisson',
  capacity: 16
} as const;

it('GPUCellTableCompare validates its props', () => {
  expectThrows(() => ({measure: 'median' as 'sum'}), /measure/);
  expectThrows(() => ({zScore: 'x' as 'poisson'}), /zScore/);
  expectThrows(() => ({sumScale: 0}), /sumScale/);
  expectThrows(graph => ({measure: 'sum', before: createTable(graph, 4, false)}), /needs sums/);
  expectThrows(
    graph => ({
      output: {
        ...createProps(graph).output,
        delta: createTransientView(graph, 'short', 'float32', 3)
      }
    }),
    /needs output.cells.length rows/
  );
  expectThrows(graph => {
    const before = createTable(graph, 8);
    return {
      before,
      output: {...createProps(graph).output, cells: before.cells}
    };
  }, /outputs must not share/);
  expectThrows(
    graph => ({
      output: {
        ...createProps(graph).output,
        cells: createTransientView(graph, 'c', 'uint32', 8) as never
      }
    }),
    /uint32x2/
  );
});

it('GPUCellTableCompare names nodes deterministically and stays within 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, {measure: 'sum', zScore: 'standardized'});
  const nodes = new GPUCellTableCompare({
    ...props,
    id: 'cmp'
  }).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'cmp-flags',
    'cmp-scan-level-0-scan',
    'cmp-merge',
    'cmp-measures',
    'cmp-statistics-sums',
    'cmp-statistics-squares',
    'cmp-statistics',
    'cmp-derived',
    'cmp-tail',
    'cmp-publish'
  ]);
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  // The z-score reduction is three kernels, with no atomics, one workgroup tile per 256 slots.
  const statistics = sources.filter(({id}) => id.startsWith('cmp-statistics'));
  expect(statistics.map(({id}) => id)).toEqual([
    'cmp-statistics-sums',
    'cmp-statistics-squares',
    'cmp-statistics'
  ]);
  for (const {id, source} of statistics) {
    expect(source, id).not.toContain('atomic');
    // Capacity 8 needs one workgroup.
    expect(source, id).toContain('const GROUPS: u32 = 1u;');
  }
  device.destroy();
});

it('GPUCellTableCompare spreads the z-score reduction over workgroups for large capacities', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const capacity = 3000;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `large-${serial++}`, format, length);
  const props = createProps(graph, {
    zScore: 'standardized',
    output: {
      cells: view('uint32x2', capacity),
      presence: view('uint32', capacity),
      delta: view('float32', capacity),
      zScore: view('float32', capacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  });
  const nodes = new GPUCellTableCompare({...props, id: 'big'}).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes).filter(({id}) =>
    id.startsWith('big-statistics')
  );
  expect(sources).toHaveLength(3);
  // ceil(3000 / 256) = 12 workgroups.
  for (const {id, source} of sources) {
    expect(source, id).toContain('const GROUPS: u32 = 12u;');
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
  device.destroy();
});

it('compareCellTablesOnCPU builds the outer join', () => {
  const result = compareCellTablesOnCPU(
    [cell(1n, 4), cell(5n, 2), cell(9n, 7)],
    [cell(2n, 3), cell(5n, 6), cell(9n << 32n, 1)],
    options
  );
  expect(result.rows.map(row => [row.key, row.presence, row.before, row.after, row.delta])).toEqual(
    [
      [1n, 1, 4, 0, -4],
      [2n, 2, 0, 3, 3],
      [5n, 3, 2, 6, 4],
      [9n, 1, 7, 0, -7],
      [9n << 32n, 2, 0, 1, 1]
    ]
  );
  expect(result.total).toBe(5);
  expect(result.overflow).toBe(false);
  expect(result.rows[0].ratio).toBe(0);
  expect(result.rows[1].ratio).toBeNaN();
  expect(result.rows[1].percentChange).toBeNaN();
  expect(result.rows[2].percentChange).toBe(200);
  expect(result.rows[2].zScore).toBe(Math.fround(4 / Math.sqrt(8)));
});

it('compareCellTablesOnCPU bounds the union and flags overflow', () => {
  const result = compareCellTablesOnCPU([cell(1n, 1), cell(3n, 1)], [cell(2n, 1), cell(4n, 1)], {
    ...options,
    capacity: 3
  });
  expect(result.rows.map(row => row.key)).toEqual([1n, 2n, 3n]);
  expect([result.total, result.overflow]).toEqual([4, true]);
  expect(compareCellTablesOnCPU([], [], options)).toEqual({
    rows: [],
    total: 0,
    overflow: false
  });
});

it('compareCellTablesOnCPU computes sum deltas exactly in 64-bit fixed point', () => {
  // Two sums that differ by 1 fixed-point unit at 2^40 magnitude: f32 subtraction would give 0.
  const big = (1n << 40n) + 1n;
  const result = compareCellTablesOnCPU([cell(1n, 1, 1n << 40n)], [cell(1n, 1, big)], {
    ...options,
    measure: 'sum'
  });
  expect(result.rows[0].delta).toBe(Math.fround(1 / 65536));
  expect(result.rows[0].before).toBe(2 ** 40 / 65536);
  expect(convertInt64ToFloat32(-5n)).toBe(-5);
  expect(convertInt64ToFloat32((1n << 33n) + 3n)).toBe(Math.fround(2 ** 33 + 3));
});

it('compareCellTablesOnCPU standardizes deltas with the population standard deviation', () => {
  const before = [cell(1n, 0), cell(2n, 0), cell(3n, 0), cell(4n, 0)];
  const after = [cell(1n, 1), cell(2n, 2), cell(3n, 3), cell(4n, 4)];
  const result = compareCellTablesOnCPU(before, after, {
    ...options,
    zScore: 'standardized'
  });
  const sd = Math.sqrt(1.25);
  expect(result.rows[0].zScore).toBeCloseTo((1 - 2.5) / sd, 5);
  expect(result.rows[3].zScore).toBeCloseTo((4 - 2.5) / sd, 5);
  const same = compareCellTablesOnCPU([cell(1n, 1), cell(2n, 1)], [cell(1n, 1), cell(2n, 1)], {
    ...options,
    zScore: 'standardized'
  });
  expect(same.rows.map(row => row.zScore)).toEqual([0, 0]);
  expect(sumFixedOrder(Array.from({length: 1000}, (_, i) => i))).toBe(499500);
});

it('GPUCellTableCompare rejects overlapping output bindings even for logically disjoint views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'shared-outputs', 'uint32', 256);
    const view = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'float32', length: 8, byteOffset});
    for (const byteOffset of [0, 32, 252]) {
      expect(
        () =>
          new GPUCellTableCompare({
            ...props,
            output: {...props.output, before: view(0), after: view(byteOffset)}
          })
      ).toThrow(/overlap/);
    }
    expect(
      () =>
        new GPUCellTableCompare({
          ...props,
          output: {...props.output, before: view(0), after: view(256)}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('GPUCellTableCompare rejects oversized aligned bindings and derived scratch without mutating the graph', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const testCase of ['before', 'output', 'bounds'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph);
      const backing = createTransientView(graph, 'prefixed-keys', 'uint32', 128);
      const oversized = graph.createDataView(backing.buffer, {
        format: 'uint32x2',
        length: 8,
        byteOffset: 4
      });
      if (testCase === 'before') props.before = {...createTable(graph, 8), cells: oversized};
      if (testCase === 'output') props.output.cells = oversized;
      if (testCase === 'bounds') {
        props.before = createTable(graph, 1);
        props.after = createTable(graph, 1);
        props.output = {
          cells: createTransientView(graph, 'small-output', 'uint32x2', 1),
          status: props.output.status
        };
      }
      const contributor = new GPUCellTableCompare(props);
      Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
        value: testCase === 'bounds' ? 12 : 64
      });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      const add = vi.spyOn(graph, 'add');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        testCase === 'bounds' ? /match bounds scratch/ : /limit|byteLength/
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

it('GPUCellTableCompare bounds two-word row addressing and union counts at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(
      () =>
        new GPUCellTableCompare(
          createProps(graph, {id: 'checked-compare', before: createTable(graph, 2 ** 31)})
        )
    ).toThrow(/^checked-compare before.cells.length must be an integer in/);
    expect(
      () =>
        new GPUCellTableCompare(
          createProps(graph, {id: 'checked-compare', after: createTable(graph, 2 ** 31)})
        )
    ).toThrow(/^checked-compare after.cells.length must be an integer in/);
  } finally {
    device.destroy();
  }
});

it('allows disjoint packed bytes for table keys and measures written by separate dispatches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'packed-columns', 'uint32', 48);
    const cells = graph.createDataView(backing.buffer, {
      format: 'uint32x2',
      length: 8,
      byteOffset: 4
    });
    const delta = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 8,
      byteOffset: 68
    });
    const contributor = new GPUCellTableCompare({
      ...props,
      output: {...props.output, cells, delta}
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('stages packed measures before derived writes to satisfy physical storage usage scopes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, 'packed-numerics', 'float32', 200);
    const column = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {
        format: 'float32',
        length: 8,
        byteOffset
      });
    const nodes = new GPUCellTableCompare({
      ...props,
      measure: 'sum',
      sumScale: 2 ** 126,
      output: {
        ...props.output,
        before: column(4),
        after: column(36),
        delta: column(68),
        // Each measure is published separately; the three derived outputs are co-bound.
        ratio: column(100),
        percentChange: column(356),
        zScore: column(612)
      }
    }).getCommandNodes(graph);
    expectCellStorageScopes(nodes);
    expect(nodes.filter(node => /-publish-(before|after|delta)$/.test(node.id))).toHaveLength(3);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expectCellBarrierEntry(kernel.source);
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      if (kernel.source.includes('fn getComparePercent('))
        expect(kernel.source).toContain(
          'getScaledQuotient(parts.fract * 100.0, divisor, 1.0, parts.exp)'
        );
      if (kernel.source.includes('beforeOut[beforeOutOffset + index] ='))
        expect(kernel.source).toContain(
          'getScaledQuotient(cellI64ToF32(beforeValue), SUM_SCALE, 1.0, 0)'
        );
    }
  } finally {
    device.destroy();
  }
});

it('count comparison ignores unused sum columns without storage usage', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const unused = graph.createTransientBuffer({
      id: 'unused-before-sums',
      byteLength: 64,
      usage: Buffer.COPY_DST
    });
    props.before.sums = graph.createDataView(unused, {
      format: 'uint32x2',
      length: props.before.cells.length,
      byteOffset: 4
    });
    expect(() =>
      new GPUCellTableCompare({...props, measure: 'count'}).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
