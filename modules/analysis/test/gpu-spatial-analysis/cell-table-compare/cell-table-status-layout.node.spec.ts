// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCellAggregation,
  GPUCellPyramid,
  GPUCellRollup
} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {GPUCellTableCompare} from '../../../src/gpu-spatial-analysis/cell-table-compare/index';
import {createCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {createCellRollup} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-rollup';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it.each([
  'aggregation',
  'rollup'
] as const)('%s composition factory preserves inherited props', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const source = createTable(graph, 'source');
    const output = createTable(graph, 'output');
    const props = Object.assign(
      Object.create({
        family: 'quadbin',
        resolution: 0,
        sourceResolution: 1,
        source,
        cells: source.cells,
        values: createTransientView(graph, 'values', 'float32', 1),
        output
      }),
      {extraCounts: []}
    );
    expect(() =>
      (kind === 'aggregation'
        ? createCellAggregation(props)
        : createCellRollup(props)
      ).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

function createTable(graph: GPUCommandGraph, name: string, capacity = 1) {
  return {
    cells: createTransientView(graph, `${name}-cells`, 'uint32x2', capacity),
    counts: createTransientView(graph, `${name}-counts`, 'uint32', capacity),
    sums: createTransientView(graph, `${name}-sums`, 'uint32x2', capacity),
    status: {
      count: createTransientView(graph, `${name}-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${name}-overflow`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `${name}-candidate`, 'uint32', 1)
    }
  };
}

function createContributor(
  kind: 'aggregation' | 'rollup' | 'pyramid' | 'comparison',
  graph: GPUCommandGraph,
  output: ReturnType<typeof createTable>
) {
  const source = createTable(graph, `${output.cells.buffer.id}-${kind}-input`);
  const values = createTransientView(
    graph,
    `${output.cells.buffer.id}-${kind}-values`,
    'float32',
    1
  );
  if (kind === 'comparison')
    return new GPUCellTableCompare({before: source, after: source, output});
  if (kind === 'rollup')
    return new GPUCellRollup({
      family: 'quadbin',
      sourceResolution: 1,
      resolution: 0,
      source,
      output
    });
  if (kind === 'pyramid')
    return new GPUCellPyramid({
      family: 'quadbin',
      cells: source.cells,
      values,
      resolutions: [1, 0],
      output: {levels: [createTable(graph, `${output.cells.buffer.id}-finest`), output]}
    });
  return new GPUCellAggregation({
    family: 'quadbin',
    resolution: 0,
    cells: source.cells,
    values,
    output
  });
}

it('refuses overlapping aligned count/sum bindings for rollup and coarse pyramid but permits raw aggregation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const packed = createTransientView(graph, 'packed-count-sum', 'uint32', 4);
  const output = createTable(graph, 'output');
  output.counts = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  output.sums = graph.createDataView(packed.buffer, {format: 'uint32x2', length: 1, byteOffset: 8});
  try {
    expect(() =>
      createContributor('aggregation', graph, output).getCommandNodes(graph)
    ).not.toThrow();
    expect(() => createContributor('rollup', graph, output)).toThrow(
      /counts.*sums.*aligned binding.*must not overlap/
    );
    expect(() => createContributor('pyramid', graph, output)).toThrow(
      /counts.*sums.*aligned binding.*must not overlap/
    );
  } finally {
    device.destroy();
  }
});

it.each([
  'aggregation',
  'rollup',
  'pyramid',
  'comparison'
] as const)('%s ignores table and status metadata', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output = {...createTable(graph, 'metadata'), label: 'trial table'};
    Object.assign(output.status, {label: 'trial status'});
    expect(() => createContributor(kind, graph, output).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'aggregation',
  'rollup',
  'pyramid',
  'comparison'
] as const)('%s rejects inherited same-word loss ports', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const inherited = createTable(graph, 'inherited');
    inherited.status = Object.assign(
      Object.create({candidateOverflow: inherited.status.overflow}),
      {
        count: inherited.status.count,
        overflow: inherited.status.overflow,
        label: 'trial status'
      }
    );
    expect(() => createContributor(kind, graph, inherited)).toThrow(/logical.*must not overlap/);
  } finally {
    device.destroy();
  }
});

it.each([
  'aggregation',
  'rollup',
  'pyramid',
  'comparison'
] as const)('%s captures inherited getters once and keeps declared nodes after caller mutation', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const baselineGraph = new GPUCommandGraph(device);
  const output = createTable(graph, 'output');
  const candidate = output.status.candidateOverflow;
  let reads = 0;
  Object.setPrototypeOf(output.status, {
    get candidateOverflow() {
      reads++;
      return candidate;
    }
  });
  delete (output.status as Partial<typeof output.status>).candidateOverflow;
  try {
    const contributor = createContributor(kind, graph, output);
    expect(reads).toBe(1);
    const baseline = createContributor(kind, baselineGraph, createTable(baselineGraph, 'output'));
    const before = getGPUCommandNodeKernelSources(baseline.getCommandNodes(baselineGraph));
    output.cells = createTransientView(graph, 'changed-cells', 'uint32x2', 2);
    output.status.overflow = output.status.count;
    expect(getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))).toEqual(before);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it.each([
  'comparison',
  'rollup',
  'aggregation',
  'tiled aggregation',
  'pyramid'
] as const)('%s accepts adjacent loss words but rejects the same logical word', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const table = (name: string, candidateByteOffset = 4) => {
    const flags = createTransientView(graph, `${name}-flags`, 'uint32', 3);
    return {
      cells: createTransientView(graph, `${name}-cells`, 'uint32x2', 4),
      counts: createTransientView(graph, `${name}-counts`, 'uint32', 4),
      status: {
        count: createTransientView(graph, `${name}-count`, 'uint32', 1),
        overflow: graph.createDataView(flags.buffer, {format: 'uint32', length: 1}),
        candidateOverflow: graph.createDataView(flags.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: candidateByteOffset
        })
      }
    };
  };
  const source = table('source');
  const create = (name: string, candidateByteOffset: number, adjacentCount = false) => {
    const output = table(name, candidateByteOffset);
    if (adjacentCount)
      output.status.count = graph.createDataView(output.status.overflow.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 8
      });
    const common = {family: 'quadbin' as const, resolution: 1, output};
    if (kind === 'comparison')
      return new GPUCellTableCompare({
        measure: 'count',
        before: source,
        after: source,
        output
      });
    if (kind === 'rollup') return new GPUCellRollup({...common, sourceResolution: 1, source});
    if (kind === 'pyramid')
      return new GPUCellPyramid({
        family: 'quadbin',
        cells: source.cells,
        resolutions: [1, 0],
        output: {levels: [output, table(`${name}-coarse`, candidateByteOffset)]}
      });
    return new GPUCellAggregation({
      ...common,
      cells: source.cells,
      ...(kind === 'tiled aggregation' ? {scratchTileSize: 2} : {})
    });
  };
  try {
    expect(() => create('adjacent', 4).getCommandNodes(graph)).not.toThrow();
    expect(() => create('aliased', 0)).toThrow(/logical.*must not overlap/);
    expect(() => create('co-bound', 4, true)).toThrow(/aligned binding.*must not overlap/);
  } finally {
    device.destroy();
  }
});
