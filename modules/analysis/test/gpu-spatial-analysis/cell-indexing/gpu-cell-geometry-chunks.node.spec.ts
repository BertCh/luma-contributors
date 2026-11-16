// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {addCellChunkNodes} from '../cell-topology/gpu-cell-chunks.test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
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

function createFields(graph: GPUCommandGraph, lengths: number[], stride = 4) {
  const cells = lengths.map((length, chunkIndex) => {
    const backing = createTransientView(graph, `cells-${chunkIndex}`, 'uint32', length * 2 + 67);
    return graph.createDataView(backing.buffer, {
      format: 'uint32x2',
      length,
      byteOffset: [4, 260, 12][chunkIndex % 3]
    });
  });
  const createOutput = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    multiplier = 1
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const backing = createTransientView(
          graph,
          `${name}-${chunkIndex}`,
          'uint32',
          length * multiplier * (format === 'float32x2' ? 2 : 1) + 67
        );
        return graph.createDataView(backing.buffer, {
          format,
          length: length * multiplier,
          byteOffset: [12, 4, 260][chunkIndex % 3]
        });
      })
    );
  return {
    cells: createVectorView('cells', 'uint32x2', cells),
    output: {
      centers: createOutput('centers', 'float32x2'),
      boundaries: createOutput('boundaries', 'float32x2', stride),
      vertexCounts: createOutput('vertexCounts', 'uint32'),
      validity: createOutput('validity', 'uint32')
    }
  };
}

it('scopes chunk resources to the contributor id in a shared graph', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 4, 2]);
  const nodes = ['first', 'second'].flatMap(id =>
    addCellChunkNodes(graph, new GPUCellGeometry({id, family: 'quadbin', ...fields}))
  );
  expect(nodes.map(node => node.id)).toEqual([
    'first-chunks-upload',
    'first-geometry-0',
    'first-geometry-1',
    'first-geometry-2',
    'second-chunks-upload',
    'second-geometry-0',
    'second-geometry-1',
    'second-geometry-2'
  ]);
});

it('checks parameter maxBufferSize before declaring resources', () => {
  const limitedDevice = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(limitedDevice);
    const contributor = new GPUCellGeometry({family: 'quadbin', ...createFields(graph, [1, 1, 1])});
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

it('rejects unrepresentable maximumVertexCount literals at construction', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1]);
  for (const maximumVertexCount of [NaN, Infinity, -Infinity, 1e100, 2 ** 32, 4.5, -1])
    expect(
      () =>
        new GPUCellGeometry({
          id: 'numeric-geometry',
          family: 'quadbin',
          maximumVertexCount,
          ...fields
        })
    ).toThrow(/^numeric-geometry maximumVertexCount/);
});

it('validates matching row topology, whole boundary records and packed chunk formats', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 4, 2]);
  expect(new GPUCellGeometry({family: 'quadbin', ...fields}).estimatedCommandNodeCount).toBe(4);
  const reversed = createVectorView(
    'reversed',
    'float32x2',
    [...fields.output.centers.data].reverse()
  );
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        ...fields,
        output: {...fields.output, centers: reversed}
      })
  ).toThrow(/centers chunk 0.*topology/);
  const splitRecord = createVectorView('split-record', 'float32x2', [
    fields.output.boundaries.data[0],
    graph.createDataView(fields.output.boundaries.data[1].buffer, {
      format: 'float32x2',
      length: 15
    }),
    graph.createDataView(fields.output.boundaries.data[2].buffer, {format: 'float32x2', length: 9})
  ]);
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        ...fields,
        output: {...fields.output, boundaries: splitRecord}
      })
  ).toThrow(/boundaries chunk 1 length.*maximumVertexCount/);
  const strided = graph.createDataView(fields.cells.data[1].buffer, {
    format: 'uint32x2',
    length: 4,
    byteStride: 12
  });
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        ...fields,
        cells: createVectorView('strided', 'uint32x2', [
          fields.cells.data[0],
          strided,
          fields.cells.data[2]
        ])
      })
  ).toThrow(/cells chunk 1 must be packed/);
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        cells: createTransientView(graph, 'wrong-format', 'uint32', 7) as never,
        output: {centers: createTransientView(graph, 'centers-packed', 'float32x2', 7)}
      })
  ).toThrow(/cells chunk 0.*uint32x2/);
});

it('rejects writable overlaps across fields and chunks at construction', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [2, 2, 2]);
  const repeated = createVectorView('repeated', 'float32x2', [
    fields.output.centers.data[0],
    fields.output.centers.data[0],
    fields.output.centers.data[2]
  ]);
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        ...fields,
        output: {...fields.output, centers: repeated}
      })
  ).toThrow(/centers chunk 0 and centers chunk 1.*overlap/);
  const validity = createVectorView(
    'aliased',
    'uint32',
    fields.output.centers.data.map(view =>
      graph.createDataView(view.buffer, {
        format: 'uint32',
        length: 2,
        byteOffset: view.byteOffset + 4
      })
    )
  );
  expect(
    () => new GPUCellGeometry({family: 'quadbin', ...fields, output: {...fields.output, validity}})
  ).toThrow(/centers chunk 0 and validity chunk 0.*overlap/);
});

it('checks later aligned boundary bindings before declaring any resource or node', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 1_048_576], 16);
  const contributor = new GPUCellGeometry({family: 'quadbin', maximumVertexCount: 16, ...fields});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  // The logical 128 MiB output fits; its four-byte aligned-binding prefix does not.
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /boundaries chunk 1 needs 134217732 bytes.*134217728/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('accepts 30 million rows through bindings that each fit the device limit', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [10_000_000, 10_000_000, 10_000_000]);
  const contributor = new GPUCellGeometry({
    family: 'quadbin',
    cells: fields.cells,
    output: {centers: fields.output.centers}
  });
  expect(addCellChunkNodes(graph, contributor)).toHaveLength(contributor.estimatedCommandNodeCount);
});

it('skips empty chunks and enforces the shared command-node budget', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [0, 1, 0, 3]);
  const contributor = new GPUCellGeometry({family: 'quadbin', ...fields});
  expect(addCellChunkNodes(graph, contributor)).toHaveLength(3);
  const cells = fields.cells.data[1];
  const centers = fields.output.centers.data[1];
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        cells: createVectorView('many', 'uint32x2', new Array(32768).fill(cells)),
        output: {
          centers: createVectorView(
            'many-centers',
            'float32x2',
            Array.from({length: 32768}, (_, index) =>
              createTransientView(graph, `center-${index}`, 'float32x2', centers.length)
            )
          )
        }
      })
  ).toThrow(/estimated command node count.*32768/);
});

it('keeps one shader and pipeline as unequal, unaligned chunks grow from 1 to 100', () => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const fields = createFields(
      graph,
      Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3])
    );
    const nodes = addCellChunkNodes(graph, new GPUCellGeometry({family: 'quadbin', ...fields}));
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
  'quadbin',
  'quadkey',
  'geohash',
  's2',
  'h3',
  'a5'
] as const)('checks every generated %s shader identifier against reserved words', family => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 4, 2], 10);
  const nodes = addCellChunkNodes(
    graph,
    new GPUCellGeometry({family, maximumVertexCount: 10, ...fields})
  );
  for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
    const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
    expect(
      (withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
        WGSL_RESERVED_WORDS.has(identifier)
      )
    ).toEqual([]);
  }
});

it('rejects logically disjoint outputs whose aligned binding prefixes overlap', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 1, 1]);
  const validity = createVectorView(
    'aligned-validity',
    'uint32',
    fields.output.centers.data.map(view =>
      graph.createDataView(view.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: view.byteOffset + 8
      })
    )
  );
  expect(
    () =>
      new GPUCellGeometry({
        family: 'quadbin',
        cells: fields.cells,
        output: {centers: fields.output.centers, validity}
      })
  ).toThrow(/aligned binding.*overlap/);
});

it('permits consecutive center chunks in the same aligned storage window', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [1, 1, 1]);
  const backing = createTransientView(graph, 'consecutive-centers', 'float32x2', 3);
  const centers = createVectorView(
    'centers',
    'float32x2',
    [0, 1, 2].map(row =>
      graph.createDataView(backing.buffer, {format: 'float32x2', length: 1, byteOffset: row * 8})
    )
  );
  expect(
    addCellChunkNodes(
      graph,
      new GPUCellGeometry({family: 'quadbin', ...fields, output: {...fields.output, centers}})
    )
  ).toHaveLength(4);
});
