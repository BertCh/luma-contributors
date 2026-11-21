// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {addCellChunkNodes} from './gpu-cell-chunks.test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {GPUCellGridPath} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {
  getCellTopologyStride,
  GPUCellTopology
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {GPUHealpixPointToCell} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell';
import {CELL_CHUNK_CASES, createCellChunkFields} from './gpu-cell-chunks.test-utils';
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

it.each(CELL_CHUNK_CASES)('$name scopes every chunk resource to its contributor id', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 4, 2], testCase.stride);
  const nodes = ['first', 'second'].flatMap(id => {
    const contributorNodes = addCellChunkNodes(graph, testCase.create(fields, id));
    expect(contributorNodes.every(node => node.id.startsWith(`${id}-`))).toBe(true);
    return contributorNodes;
  });
  expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
});

it.each(
  CELL_CHUNK_CASES
)('$name pre-checks the complete parameter allocation against maxBufferSize', testCase => {
  const limitedDevice = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(limitedDevice);
    const contributor = testCase.create(createCellChunkFields(graph, [1, 1, 1], testCase.stride));
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

it('rejects unrepresentable numeric shader props at construction', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1], 1);
  for (const value of [NaN, Infinity, -Infinity, 1e100, 2 ** 32, 1.5, -1]) {
    const topologyProps = {
      id: 'numeric-topology',
      family: 's2' as const,
      cells: fields.cells,
      output: {cells: fields.outputCells}
    };
    for (const operation of [
      {type: 'disk' as const, k: value},
      {type: 'ring' as const, k: value},
      {type: 'parent' as const, resolution: value},
      {type: 'children' as const, inputResolution: 0, resolution: value},
      {type: 'children' as const, inputResolution: value, resolution: 0}
    ])
      expect(() => new GPUCellTopology({...topologyProps, operation})).toThrow(
        /^numeric-topology /
      );
    expect(
      () =>
        new GPUCellGridPath({
          id: 'numeric-path',
          family: 'h3',
          origins: fields.cells,
          destinations: fields.destinations,
          maximumPathLength: value,
          output: {cells: fields.outputCells}
        })
    ).toThrow(/^numeric-path maximumPathLength/);
    expect(
      () =>
        new GPUS2CellHierarchy({
          id: 'numeric-hierarchy',
          cells: fields.cells,
          operation: 'parent',
          level: value,
          output: {cells: fields.outputCells, status: fields.status}
        })
    ).toThrow(/^numeric-hierarchy level/);
    expect(
      () =>
        new GPUHealpixPointToCell({
          id: 'numeric-healpix',
          order: value,
          positions: fields.positions,
          output: {cells: fields.outputCells}
        })
    ).toThrow(/^numeric-healpix order/);
  }
});

it.each(
  CELL_CHUNK_CASES
)('$name validates row topology and whole per-row output records', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 4, 2], testCase.stride);
  expect(addCellChunkNodes(graph, testCase.create(fields))).toHaveLength(4);
  const mismatch = {...fields};
  if (testCase.input === 'cells')
    mismatch.cells = createVectorView('mismatch', 'uint32x2', [...fields.cells.data].reverse());
  else
    mismatch.positions = createVectorView(
      'mismatch',
      'float32x2',
      [...fields.positions.data].reverse()
    );
  expect(() => testCase.create(mismatch)).toThrow(/topology|length must equal/);
  if (testCase.outputs.includes('outputCells') && testCase.stride > 1) {
    const split = {
      ...fields,
      outputCells: createVectorView('split-record', 'uint32x2', [
        fields.outputCells.data[0],
        graph.createDataView(fields.outputCells.data[1].buffer, {
          format: 'uint32x2',
          length: 4 * testCase.stride - 1
        }),
        graph.createDataView(fields.outputCells.data[2].buffer, {
          format: 'uint32x2',
          length: 2 * testCase.stride + 1
        })
      ])
    };
    expect(() => testCase.create(split)).toThrow(/chunk 1 length must equal/);
  }
});

it.each(CELL_CHUNK_CASES)('$name rejects writable chunk aliases at construction', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [2, 2, 2], testCase.stride);
  const outputName = testCase.outputs[0];
  if (outputName === 'measures')
    fields.measures = createVectorView('repeat', 'float32x2', [
      fields.measures.data[0],
      fields.measures.data[0],
      fields.measures.data[2]
    ]);
  else
    fields.outputCells = createVectorView('repeat', 'uint32x2', [
      fields.outputCells.data[0],
      fields.outputCells.data[0],
      fields.outputCells.data[2]
    ]);
  expect(() => testCase.create(fields)).toThrow(/chunk 0 and .*chunk 1.*overlap/);
});

it.each(
  CELL_CHUNK_CASES
)('$name pre-checks later aligned bindings without graph mutation', testCase => {
  const limitedDevice = createNullWebGPUDevice();
  Object.defineProperty(limitedDevice.limits, 'maxStorageBufferBindingSize', {value: 64});
  try {
    const graph = new GPUCommandGraph(limitedDevice);
    const contributor = testCase.create(createCellChunkFields(graph, [1, 20, 1], testCase.stride));
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/chunk 1 needs 164 bytes.*limit 64/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    limitedDevice.destroy();
  }
});

it.each(
  CELL_CHUNK_CASES
)('$name keeps shader and pipeline counts constant for unequal unaligned chunks', testCase => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 20]) {
    const graph = new GPUCommandGraph(device);
    const fields = createCellChunkFields(
      graph,
      Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
      testCase.stride
    );
    const nodes = addCellChunkNodes(graph, testCase.create(fields));
    const invocationsPerRow = testCase.name.startsWith('GPUH3CellIdentities')
      ? 6
      : testCase.name === 'GPUCellTopology children'
        ? 4
        : 1;
    expect(nodes.slice(1).map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      Array.from({length: chunkCount}, (_, index) =>
        Math.ceil(([1, 257, 513][index % 3] * invocationsPerRow) / 256)
      )
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

it.each(CELL_CHUNK_CASES)('$name skips empty chunks', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [0, 1, 0, 3], testCase.stride);
  expect(addCellChunkNodes(graph, testCase.create(fields))).toHaveLength(3);
});

it.each(
  CELL_CHUNK_CASES
)('$name checks every shader identifier against the full reserved list', testCase => {
  const graph = new GPUCommandGraph(device);
  const nodes = addCellChunkNodes(
    graph,
    testCase.create(createCellChunkFields(graph, [1, 4, 2], testCase.stride))
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

it.each(CELL_CHUNK_CASES)('$name accepts 30 million rows in bounded bindings', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, new Array(30).fill(1_000_000), testCase.stride);
  const contributor = testCase.create(fields);
  expect(addCellChunkNodes(graph, contributor)).toHaveLength(31);
});

it('declares offsets only for bound optional fields, on every spatial family', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 4, 2], 1);
  const contributors = [
    new GPUCellGridPath({
      family: 'h3',
      origins: fields.cells,
      destinations: fields.destinations,
      output: {distances: fields.distances}
    }),
    ...(['h3', 's2', 'quadbin'] as const).map(
      family =>
        new GPUCellTopology({
          id: family,
          family,
          operation: {type: 'parent', resolution: 0},
          cells: fields.cells,
          output: {cells: fields.outputCells}
        })
    ),
    new GPUHealpixPointToCell({
      order: 0,
      positions: fields.positions,
      output: {cells: fields.outputCells}
    })
  ];
  for (const contributor of contributors) {
    const nodes = addCellChunkNodes(graph, contributor);
    expect(getParsedKernelSources(nodes).size).toBe(1);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
      expect(
        (withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
          WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
      const names = new Set(
        [...source.matchAll(/var<storage,\s*\w+>\s*(\w+):/g)].map(match => match[1])
      );
      for (const match of source.matchAll(/arrayLength\(&([A-Za-z_][A-Za-z_0-9]*)\)/g))
        expect(names.has(match[1]), match[1]).toBe(true);
    }
  }
});

it.each([
  false,
  true
])('adds packed grid-path distance nodes and validates their format (combined=%s)', combined => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1], 4);
  const props = {
    family: 'h3' as const,
    origins: fields.cells.data[0],
    destinations: fields.destinations.data[0],
    maximumPathLength: 4,
    output: {
      distances: fields.distances.data[0],
      ...(combined ? {cells: fields.outputCells.data[0], counts: fields.counts.data[0]} : {})
    }
  };
  expect(addCellChunkNodes(graph, new GPUCellGridPath(props))).toHaveLength(2);
  expect(
    () =>
      new GPUCellGridPath({
        ...props,
        output: {
          ...props.output,
          distances: createTransientView(graph, 'bad-distance-format', 'float32', 1) as never
        }
      })
  ).toThrow(/distances chunk 0.*uint32/);
});

it.each(
  CELL_CHUNK_CASES
)('$name permits consecutive output chunks in one alignment window', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 1, 1], testCase.stride);
  if (testCase.outputs[0] === 'measures') {
    const backing = createTransientView(graph, 'consecutive', 'float32x2', 3);
    fields.measures = createVectorView(
      'consecutive',
      'float32x2',
      [0, 1, 2].map(row =>
        graph.createDataView(backing.buffer, {format: 'float32x2', length: 1, byteOffset: row * 8})
      )
    );
  } else {
    const backing = createTransientView(graph, 'consecutive', 'uint32x2', 3 * testCase.stride);
    fields.outputCells = createVectorView(
      'consecutive',
      'uint32x2',
      [0, 1, 2].map(row =>
        graph.createDataView(backing.buffer, {
          format: 'uint32x2',
          length: testCase.stride,
          byteOffset: row * testCase.stride * 8
        })
      )
    );
  }
  expect(addCellChunkNodes(graph, testCase.create(fields))).toHaveLength(4);
});

it.each(
  CELL_CHUNK_CASES.filter(testCase => testCase.outputs.length > 1)
)('$name rejects aligned output aliases despite disjoint logical bytes', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 1, 1], testCase.stride);
  const primary = testCase.outputs[0] === 'measures' ? fields.measures : fields.outputCells;
  const secondary = testCase.outputs.includes('counts')
    ? 'counts'
    : testCase.outputs.includes('status')
      ? 'status'
      : 'validity';
  fields[secondary] = createVectorView(
    'aligned-alias',
    'uint32',
    primary.data.map(view =>
      graph.createDataView(view.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: view.byteOffset + view.length * 8
      })
    )
  );
  expect(() => testCase.create(fields)).toThrow(/aligned binding.*overlap/);
});

it('checks all added topology, hierarchy and high-order UNIQ branches against the full reserved list', () => {
  const check = (nodes: ReturnType<typeof addCellChunkNodes>) => {
    expect(getParsedKernelSources(nodes).size).toBe(1);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(Object.keys(constants)).toEqual([]);
      const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
      expect(
        (withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
          WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
    }
  };
  for (const family of ['h3', 's2', 'quadbin'] as const) {
    for (const operation of [
      {type: 'disk', k: 1},
      {type: 'ring', k: 1},
      {type: 'children', inputResolution: 0, resolution: 1}
    ] as const) {
      const graph = new GPUCommandGraph(device);
      const stride = getCellTopologyStride(family, operation);
      const fields = createCellChunkFields(graph, [1, 257, 513], stride, undefined, stride);
      check(
        addCellChunkNodes(
          graph,
          new GPUCellTopology({
            family,
            operation,
            cells: fields.cells,
            mask: fields.mask,
            output: {
              cells: fields.outputCells,
              counts: fields.counts,
              distances: operation.type === 'disk' ? fields.distances : undefined
            }
          })
        )
      );
    }
  }
  for (const operation of ['parent', 'children', 'range', 'neighbors'] as const) {
    const graph = new GPUCommandGraph(device);
    const stride = operation === 'parent' ? 1 : operation === 'range' ? 2 : 4;
    const fields = createCellChunkFields(graph, [1, 257, 513], stride);
    check(
      addCellChunkNodes(
        graph,
        new GPUS2CellHierarchy({
          operation,
          level: 1,
          cells: fields.cells,
          output: {cells: fields.outputCells, status: fields.status}
        })
      )
    );
  }
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 257, 513], 1);
  check(
    addCellChunkNodes(
      graph,
      new GPUHealpixPointToCell({
        order: 29,
        scheme: 'uniq',
        positions: fields.positions,
        mask: fields.mask,
        output: {cells: fields.outputCells, validity: fields.validity}
      })
    )
  );
});

it('HEALPix uses the shared row kernel while preserving one shader for unequal chunks (DGGS-115)', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 4, 2], 1);
  const contributor = new GPUHealpixPointToCell({
    order: 0,
    positions: fields.positions,
    output: {cells: fields.outputCells}
  });
  const nodes = addCellChunkNodes(graph, contributor);
  expect(getParsedKernelSources(nodes).size).toBe(1);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels).toHaveLength(3);
  expect(kernels.every(kernel => kernel.source.includes('let chunkParametersOffset = 0u;'))).toBe(
    true
  );
});

it.each(
  CELL_CHUNK_CASES.filter(
    testCase => testCase.name === 'GPUCellGridPath' || testCase.name === 'GPUS2CellHierarchy'
  )
)('$name uses the shared row dispatcher for remaining one-row routes (UNFINISHED)', testCase => {
  const graph = new GPUCommandGraph(device);
  const fields = createCellChunkFields(graph, [1, 4, 2], testCase.stride);
  const nodes = addCellChunkNodes(graph, testCase.create(fields));
  expect(getParsedKernelSources(nodes).size).toBe(1);
  expect(
    getGPUCommandNodeKernelSources(nodes).every(kernel =>
      kernel.source.includes('let chunkParametersOffset = 0u;')
    )
  ).toBe(true);
});
