// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {addCellChunkNodes} from '../cell-topology/gpu-cell-chunks.test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {GPUCellMeasures} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterEach(() => vi.restoreAllMocks());
afterAll(() => device.destroy());

function createFields(graph: GPUCommandGraph, lengths: number[], stride = 10) {
  const createField = <Format extends 'uint32x2' | 'float32' | 'uint32'>(
    name: string,
    format: Format,
    multiplier = 1
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const prefix = [4, 260, 12][chunkIndex % 3];
        const backing = createTransientView(
          graph,
          `${name}-${chunkIndex}`,
          'uint32',
          length * multiplier * (format === 'uint32x2' ? 2 : 1) + 67
        );
        return graph.createDataView(backing.buffer, {
          format,
          length: length * multiplier,
          byteOffset: prefix
        });
      })
    );
  return {
    cells: createField('cells', 'uint32x2'),
    output: {
      areas: createField('areas', 'float32'),
      perimeters: createField('perimeters', 'float32'),
      edgeLengths: createField('edgeLengths', 'float32', stride),
      edgeCounts: createField('edgeCounts', 'uint32'),
      pentagons: createField('pentagons', 'uint32')
    }
  };
}

it('scopes chunk resources to the contributor id in a shared graph', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 4, 2]);
  const nodes = ['first', 'second'].flatMap(id =>
    addCellChunkNodes(
      graph,
      new GPUCellMeasures({id, family: 's2', maximumEdgeCount: 10, ...fields})
    )
  );
  expect(nodes.map(node => node.id)).toEqual([
    'first-chunks-upload',
    'first-measures-0',
    'first-measures-1',
    'first-measures-2',
    'second-chunks-upload',
    'second-measures-0',
    'second-measures-1',
    'second-measures-2'
  ]);
});

it('checks parameter maxBufferSize before declaring resources', () => {
  const limitedDevice = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(limitedDevice);
    const contributor = new GPUCellMeasures({
      family: 's2',
      maximumEdgeCount: 10,
      ...createFields(graph, [1, 1, 1])
    });
    Object.defineProperty(limitedDevice.limits, 'maxBufferSize', {value: 512});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /chunk parameter buffer byteLength.*512/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    limitedDevice.destroy();
  }
});

it('rejects unrepresentable maximumEdgeCount literals at construction', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1]);
  for (const maximumEdgeCount of [NaN, Infinity, -Infinity, 1e100, 2 ** 32, 4.5, -1])
    expect(
      () => new GPUCellMeasures({id: 'numeric-measures', family: 's2', maximumEdgeCount, ...fields})
    ).toThrow(/^numeric-measures maximumEdgeCount/);
});

it('validates matching chunks, whole fixed-stride records and every chunk format', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 4, 2]);
  const contributor = new GPUCellMeasures({family: 's2', maximumEdgeCount: 10, ...fields});
  expect(addCellChunkNodes(graph, contributor)).toHaveLength(contributor.estimatedCommandNodeCount);
  const reversed = createVectorView('reversed', 'float32', [...fields.output.areas.data].reverse());
  expect(
    () =>
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        ...fields,
        output: {...fields.output, areas: reversed}
      })
  ).toThrow(/areas chunk 0.*topology/);
  const splitRecord = createVectorView('split-record', 'float32', [
    fields.output.edgeLengths.data[0],
    graph.createDataView(fields.output.edgeLengths.data[1].buffer, {format: 'float32', length: 39}),
    graph.createDataView(fields.output.edgeLengths.data[2].buffer, {format: 'float32', length: 21})
  ]);
  expect(
    () =>
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        ...fields,
        output: {...fields.output, edgeLengths: splitRecord}
      })
  ).toThrow(/edgeLengths chunk 1 length.*maximumEdgeCount/);
  expect(
    () =>
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        ...fields,
        output: {
          ...fields.output,
          edgeCounts: createVectorView('wrong-format', 'float32', [
            ...fields.output.areas.data
          ]) as never
        }
      })
  ).toThrow(/edgeCounts chunk 0.*uint32/);
});

it('rejects overlapping outputs across fields and chunks at construction', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [2, 2, 2]);
  expect(
    () =>
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        ...fields,
        output: {...fields.output, perimeters: fields.output.areas}
      })
  ).toThrow(/areas chunk 0 and perimeters chunk 0.*overlap/);
  const repeated = createVectorView('repeated', 'float32', [
    fields.output.areas.data[0],
    fields.output.areas.data[0],
    fields.output.areas.data[2]
  ]);
  expect(
    () =>
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        ...fields,
        output: {...fields.output, areas: repeated}
      })
  ).toThrow(/areas chunk 0 and areas chunk 1.*overlap/);
});

it('checks the actual aligned range of later edge outputs before any graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 8_388_608], 4);
  const contributor = new GPUCellMeasures({family: 's2', maximumEdgeCount: 4, ...fields});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /edgeLengths chunk 1 needs 134217732 bytes.*134217728/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('accepts 30 million rows with device-sized bindings and skips empty chunks', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [10_000_000, 10_000_000, 10_000_000]);
  const contributor = new GPUCellMeasures({
    family: 's2',
    cells: fields.cells,
    output: {areas: fields.output.areas}
  });
  expect(addCellChunkNodes(graph, contributor)).toHaveLength(4);
  const emptyGraph = new GPUCommandGraph(device);
  const emptyFields = createFields(emptyGraph, [0, 1, 0, 3]);
  expect(
    addCellChunkNodes(
      emptyGraph,
      new GPUCellMeasures({
        id: 'empty',
        family: 's2',
        maximumEdgeCount: 10,
        ...emptyFields
      })
    )
  ).toHaveLength(3);
});

it.each([
  'float32',
  'fp64'
] as const)('keeps one shader and pipeline for growing unequal unaligned chunks (%s)', precision => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 20]) {
    const graph = new GPUCommandGraph(device);
    const fields = createFields(
      graph,
      Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3])
    );
    const nodes = addCellChunkNodes(
      graph,
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        precision,
        ...fields
      })
    );
    expect(nodes.slice(1).map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      Array.from({length: chunkCount}, (_, index) => Math.ceil([1, 257, 513][index % 3] / 256))
    );
    expect(getParsedKernelSources(nodes).size).toBe(1);
    expect(
      getGPUCommandNodeKernelSources(nodes).every(
        kernel => Object.keys(kernel.constants).length === 0
      )
    ).toBe(true);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
  }
  expect(keys.size).toBe(1);
});

it.each([
  'h3',
  's2',
  'quadbin'
] as const)('checks every generated %s shader identifier against the full reserved list', family => {
  for (const precision of ['float32', 'fp64'] as const) {
    const graph = new GPUCommandGraph(device);
    const nodes = addCellChunkNodes(
      graph,
      new GPUCellMeasures({
        family,
        precision,
        maximumEdgeCount: 10,
        ...createFields(graph, [1, 4, 2])
      })
    );
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
      expect(
        (withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
          WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
    }
  }
});

it('rejects logically disjoint outputs whose aligned binding prefixes overlap', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 1, 1]);
  const perimeters = createVectorView(
    'aligned-perimeters',
    'float32',
    fields.output.areas.data.map(view =>
      graph.createDataView(view.buffer, {
        format: 'float32',
        length: 1,
        byteOffset: view.byteOffset + 4
      })
    )
  );
  expect(
    () =>
      new GPUCellMeasures({
        family: 's2',
        cells: fields.cells,
        output: {areas: fields.output.areas, perimeters}
      })
  ).toThrow(/aligned binding.*overlap/);
});

it('permits consecutive area chunks in the same aligned storage window', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 1, 1]);
  const backing = createTransientView(graph, 'consecutive-areas', 'float32', 3);
  const areas = createVectorView(
    'areas',
    'float32',
    [0, 1, 2].map(row =>
      graph.createDataView(backing.buffer, {format: 'float32', length: 1, byteOffset: row * 4})
    )
  );
  expect(
    addCellChunkNodes(
      graph,
      new GPUCellMeasures({
        family: 's2',
        maximumEdgeCount: 10,
        ...fields,
        output: {...fields.output, areas}
      })
    )
  ).toHaveLength(4);
});
