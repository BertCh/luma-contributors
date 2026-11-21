// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GraphDataView,
  type GraphVectorView,
  GPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expect, expectTypeOf, it, vi} from 'vitest';
import {
  GPUPointToCell,
  type GPUPointToCellOutput,
  type GPUPointToCellProps
} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell';
import {
  GPUCellAggregation,
  createCellAggregation
} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {GPUCellLevelSelection} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-level-selection';
import {GPUCellRollup} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-rollup';
import {GPUCellPyramid} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-pyramid';
import {
  createNullWebGPUDevice,
  createVectorView,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
function createRows(graph: GPUCommandGraph, lengths = [3, 257, 771], withOffsets = false) {
  const chunks = <Format extends 'float32x2' | 'float32' | 'uint32' | 'uint32x2'>(format: Format) =>
    createVectorView(
      `chunks-${serial++}`,
      format,
      lengths.map((length, chunkIndex) => {
        const prefixRows = withOffsets ? chunkIndex % 3 : 0;
        const backing = createTransientView(
          graph,
          `chunk-${serial++}`,
          format,
          Math.max(length, 1) + prefixRows
        );
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: prefixRows * (format.endsWith('x2') ? 8 : 4)
        });
      })
    );
  return {
    positions: chunks('float32x2'),
    mask: chunks('uint32'),
    values: chunks('float32'),
    output: {cells: chunks('uint32x2'), validity: chunks('uint32')}
  };
}
function createTable(graph: GPUCommandGraph, capacity = 16) {
  return {
    cells: createTransientView(graph, `cells-${serial++}`, 'uint32x2', capacity),
    counts: createTransientView(graph, `counts-${serial++}`, 'uint32', capacity),
    sums: createTransientView(graph, `sums-${serial++}`, 'uint32x2', capacity),
    sumValues: createTransientView(graph, `sum-values-${serial++}`, 'float32', capacity),
    minimums: createTransientView(graph, `minimums-${serial++}`, 'float32', capacity),
    maximums: createTransientView(graph, `maximums-${serial++}`, 'float32', capacity),
    status: {
      count: createTransientView(graph, `count-${serial++}`, 'uint32', 1),
      overflow: createTransientView(graph, `overflow-${serial++}`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `candidate-overflow-${serial++}`, 'uint32', 1)
    }
  };
}

it('point-to-cell mirrors uneven chunks and skips empty batches without global scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {values: _values, ...rows} = createRows(graph, [0, 3, 257, 771, 0]);
  const contributor = new GPUPointToCell({family: 'quadbin', resolution: 8, ...rows});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'point-to-cell-chunks-upload',
    'point-to-cell-keys-chunk-1',
    'point-to-cell-keys-chunk-2',
    'point-to-cell-keys-chunk-3'
  ]);
  expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
  expect(getParsedKernelSources(nodes).size).toBe(3);
  expect(allocate).toHaveBeenCalledTimes(1);
  device.destroy();
});

it('chunk boundaries must match for masks, values and row-aligned outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph);
  const other = createRows(graph, [3, 258, 770]);
  const pointProps = {family: 'quadbin' as const, resolution: 8, ...rows};
  expect(() => new GPUPointToCell({...pointProps, mask: other.mask})).toThrow(/mask.*topology/);
  expect(() => new GPUPointToCell({...pointProps, output: other.output})).toThrow(
    /output.cells.*topology/
  );
  expect(() => new GPUPointToCell({...pointProps, mask: rows.mask.data[0]})).toThrow(/mask length/);
  const aggregationProps = {...pointProps, output: createTable(graph)};
  expect(() => new GPUCellAggregation({...aggregationProps, values: other.values})).toThrow(
    /values.*topology/
  );
  expect(() => new GPUCellAggregation({...aggregationProps, mask: other.mask})).toThrow(
    /mask.*topology/
  );
  device.destroy();
});

it('aggregation assigns global row offsets and preserves the existing fixed-point reductions', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [0, 3, 257, 771, 0]);
  const contributor = new GPUCellAggregation({
    ...rows,
    family: 'quadbin',
    resolution: 26,
    output: createTable(graph)
  });
  const nodes = contributor.getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(getParsedKernelSources(nodes).size).toBeGreaterThan(6);
  const keys = sources.filter(source => /-keys-chunk-/.test(source.id));
  expect(
    keys.every(source => source.source.includes('chunkParameters[chunkParametersOffset] + index'))
  ).toBe(true);
  expect(keys.every(source => !('ROW_OFFSET' in source.constants))).toBe(true);
  const copies = sources.filter(source => /-values-chunk-/.test(source.id));
  expect(copies.every(source => !('ROW_OFFSET' in source.constants))).toBe(true);
  expect(nodes[0].id).toBe('cell-aggregation-chunk-parameters-upload');
  expect(nodes.some(node => node.id.endsWith('-accumulate-sums'))).toBe(true);
  expect(nodes.some(node => node.id.endsWith('-accumulate-extremes'))).toBe(true);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  device.destroy();
});

it('aggregation never binds empty end-of-buffer chunks or gives them parameter slots', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const scratchTileSize of [undefined, 256]) {
      const graph = new GPUCommandGraph(device);
      const createSubvector = <Format extends 'float32x2' | 'float32' | 'uint32'>(
        name: string,
        format: Format
      ) => {
        const backing = createTransientView(graph, name, format, 1031);
        let rowOffset = 0;
        return createVectorView(
          name,
          format,
          [0, 1, 256, 0, 774, 0].map(length => {
            const chunk = graph.createDataView(backing.buffer, {
              format,
              length,
              byteOffset: rowOffset * backing.rowByteLength
            });
            rowOffset += length;
            return chunk;
          })
        );
      };
      const contributor = new GPUCellAggregation({
        positions: createSubvector('positions', 'float32x2'),
        values: createSubvector('values', 'float32'),
        mask: createSubvector('mask', 'uint32'),
        family: 'quadbin',
        resolution: 26,
        output: createTable(graph),
        scratchTileSize
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(contributor.usesTiledScratch).toBe(scratchTileSize !== undefined);
      for (const node of nodes) {
        for (const resource of node.resources ?? []) {
          if ('buffer' in resource && resource.buffer instanceof GraphDataView) {
            const binding = getViewBindingRange(resource.buffer);
            expect(binding.offset + binding.size, node.id).toBeLessThanOrEqual(
              resource.buffer.buffer.byteLength
            );
          }
        }
      }
      if (!scratchTileSize) {
        const prepares = nodes.filter(node => /-keys-chunk-/.test(node.id));
        expect(prepares.map(node => node.workload?.maximumInvocationCount)).toEqual([1, 256, 774]);
        expect(
          prepares.map(node => {
            const resource = node.resources!.find(
              resource =>
                'buffer' in resource &&
                resource.buffer instanceof GraphDataView &&
                resource.buffer.buffer.id.endsWith('-chunk-parameters')
            )!;
            return 'buffer' in resource && resource.buffer instanceof GraphDataView
              ? resource.buffer.byteOffset
              : undefined;
          })
        ).toEqual([0, 256, 512]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('30 million chunked rows fit scalar aggregation scratch at core device limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [7_000_001, 9_000_003, 13_999_996]);
  const contributor = new GPUCellAggregation({
    ...rows,
    family: 'quadbin',
    resolution: 26,
    output: createTable(graph)
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  for (const node of nodes) {
    for (const resource of node.resources ?? []) {
      if ('buffer' in resource && resource.buffer instanceof GraphDataView) {
        expect(getViewBindingRange(resource.buffer).size, node.id).toBeLessThanOrEqual(
          device.limits.maxStorageBufferBindingSize
        );
      }
    }
  }
  const pyramid = new GPUCellPyramid({
    ...rows,
    id: 'chunk-pyramid',
    family: 'quadbin',
    resolutions: [26, 8],
    output: {levels: [createTable(graph), createTable(graph)]}
  });
  expect(pyramid.getCommandNodes(graph).some(node => node.id.endsWith('-level-1-publish'))).toBe(
    true
  );
  device.destroy();
});

it('oversized global scratch and individual chunks fail before declaring scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [12_000_000, 12_000_000, 12_000_000]);
  const contributor = new GPUCellAggregation({
    ...rows,
    scratchMode: 'packed',
    family: 'quadbin',
    resolution: 8,
    output: createTable(graph)
  });
  const pointRows = createRows(graph, [16_777_217]);
  const point = new GPUPointToCell({...pointRows, family: 'quadbin', resolution: 8});
  // A tightly sized binding also needs its prefix up to the 256-byte aligned binding offset.
  const offsetRows = createRows(graph, [16_777_216]);
  const backing = createTransientView(graph, 'offset-positions-backing', 'float32x2', 16_777_217);
  const offsetPositions = graph.createDataView(backing.buffer, {
    format: 'float32x2',
    length: 16_777_216,
    byteOffset: 8
  });
  const offsetPoint = new GPUPointToCell({
    ...offsetRows,
    family: 'quadbin',
    resolution: 8,
    positions: createVectorView('offset-positions', 'float32x2', [offsetPositions])
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/cell-aggregation row scratch.*limit/);
  expect(() => offsetPoint.getCommandNodes(graph)).toThrow(/point-to-cell row chunk.*limit/);
  expect(() => point.getCommandNodes(graph)).toThrow(/point-to-cell row chunk.*limit/);
  expect(allocate).not.toHaveBeenCalled();
  device.destroy();
});

it('chunk membership, aliasing and command budgets are validated', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const rows = createRows(other);
  const contributor = new GPUPointToCell({...rows, family: 'quadbin', resolution: 8});
  expect(() => contributor.getCommandNodes(graph)).toThrow(/target graph/);
  expect(
    () =>
      new GPUPointToCell({
        ...rows,
        family: 'quadbin',
        resolution: 8,
        output: {...rows.output, validity: rows.mask}
      })
  ).toThrow(/outputs must not share buffers with inputs/);
  const many = createRows(graph, new Array<number>(32769).fill(1));
  expect(() => new GPUPointToCell({...many, family: 'quadbin', resolution: 8})).toThrow(
    /estimated command node count/
  );
  device.destroy();
});

it('point output chunks reject repeated and partially overlapping ranges across columns', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [256, 256]);
  const cells = rows.output.cells.data[0];
  const props = {...rows, family: 'quadbin' as const, resolution: 8};
  expect(
    () =>
      new GPUPointToCell({
        ...props,
        output: {
          ...rows.output,
          cells: createVectorView('repeated-cells', 'uint32x2', [cells, cells])
        }
      })
  ).toThrow(/output.cells chunk 0.*output.cells chunk 1.*must not overlap/);
  const validity = rows.output.validity.data[0];
  expect(
    () =>
      new GPUPointToCell({
        ...props,
        output: {
          ...rows.output,
          validity: createVectorView('repeated-validity', 'uint32', [validity, validity])
        }
      })
  ).toThrow(/output.validity chunk 0.*output.validity chunk 1.*must not overlap/);
  const partial = graph.createDataView(cells.buffer, {
    format: 'uint32',
    length: 256,
    byteOffset: 4
  });
  expect(
    () =>
      new GPUPointToCell({
        ...props,
        output: {
          ...rows.output,
          validity: createVectorView('cross-column', 'uint32', [validity, partial])
        }
      })
  ).toThrow(/output.cells chunk 0.*output.validity chunk 1.*must not overlap/);
  const shared = createTransientView(graph, 'shared-output', 'uint32x2', 512);
  const disjoint = [0, 256].map(firstRow =>
    graph.createDataView(shared.buffer, {
      format: 'uint32x2',
      length: 256,
      byteOffset: firstRow * 8
    })
  );
  expect(
    () =>
      new GPUPointToCell({
        ...props,
        output: {
          ...rows.output,
          cells: createVectorView('disjoint-cells', 'uint32x2', disjoint)
        }
      })
  ).not.toThrow();
  device.destroy();
});

it('classification and value-copy pipelines are reused as equal-sized record batches increase', () => {
  const countPipelines = (chunkCount: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const rows = createRows(graph, new Array<number>(chunkCount).fill(256));
    const contributor = new GPUCellAggregation({
      ...rows,
      family: 'quadbin',
      resolution: 26,
      output: createTable(graph)
    });
    const nodes = contributor.getCommandNodes(graph);
    const keys = getGPUCommandNodePipelineKeys(
      nodes.filter(node => /-(keys|values)-chunk-/.test(node.id)),
      {includeShaderLayout: true}
    );
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    device.destroy();
    return keys.size;
  };
  expect(countPipelines(2)).toBe(2);
  expect(countPipelines(100)).toBe(2);
  expect(countPipelines(101)).toBe(2);
});

it('every generated WGSL identifier avoids the full specification reserved-word list', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const family of ['quadbin', 'quadkey', 'geohash', 's2', 'h3'] as const) {
    for (const precision of family === 'h3'
      ? (['float32', 'double-single'] as const)
      : (['float32'] as const)) {
      const rows = createRows(graph);
      const point = new GPUPointToCell({
        ...rows,
        id: `point-${family}-${precision}`,
        family,
        precision,
        resolution: 8
      });
      checkIdentifiers(point.getCommandNodes(graph));
    }
  }
  for (const family of ['quadbin', 'h3'] as const) {
    const rows = createRows(graph);
    const aggregation = new GPUCellAggregation({
      id: `reserved-words-${family}`,
      family,
      resolution: family === 'quadbin' ? 26 : 12,
      cells: rows.output.cells,
      values: rows.values,
      mask: rows.mask,
      output: createTable(graph)
    });
    checkIdentifiers(aggregation.getCommandNodes(graph));
  }
  const rows = createRows(graph);
  const pyramid = new GPUCellPyramid({
    ...rows,
    id: 'reserved-words-pyramid',
    family: 'quadbin',
    resolutions: [26, 8, 0],
    output: {
      levels: [createTable(graph), createTable(graph), createTable(graph)],
      levelCounts: createTransientView(graph, 'reserved-level-counts', 'uint32', 3)
    }
  });
  checkIdentifiers(pyramid.getCommandNodes(graph));
  device.destroy();
});

function checkIdentifiers(nodes: readonly unknown[]): void {
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    const source = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ');
    const identifiers = source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
    expect(
      identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier)),
      kernel.id
    ).toEqual([]);
  }
}

function expectUnchangedGraphOnFailure(
  graph: GPUCommandGraph,
  run: () => unknown,
  message: RegExp
) {
  const declarations = [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'importBuffer'),
    vi.spyOn(graph, 'addComputePass'),
    vi.spyOn(graph, 'addCopyPass'),
    vi.spyOn(graph, 'addRenderPass')
  ];
  try {
    expect(run).toThrow(message);
    for (const declaration of declarations) expect(declaration).not.toHaveBeenCalled();
  } finally {
    for (const declaration of declarations) declaration.mockRestore();
  }
}

it('aligned extra-count bindings are checked before any aggregation declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph);
  const backing = createTransientView(graph, 'oversized-extra-count-backing', 'uint32', 33_554_433);
  const extraCount = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 33_554_432,
    byteOffset: 4
  });
  const aggregation = createCellAggregation({
    ...rows,
    family: 'quadbin',
    resolution: 26,
    output: createTable(graph),
    extraCounts: [extraCount]
  });
  expectUnchangedGraphOnFailure(
    graph,
    () => aggregation.getCommandNodes(graph),
    /input\/output field.*chunk.*134217732.*limit 134217728/
  );
  device.destroy();
});

it('a pyramid checks late output bindings before level slots and missing columns are declared', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph);
  const first = createTable(graph);
  const {
    sums: _sums,
    minimums: _minimums,
    maximums: _maximums,
    sumValues: _sumValues,
    ...withoutIntermediateColumns
  } = first;
  const last = createTable(graph);
  const backing = createTransientView(graph, 'oversized-late-count-backing', 'uint32', 33_554_433);
  last.status.count = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 33_554_432,
    byteOffset: 4
  });
  const pyramid = new GPUCellPyramid({
    ...rows,
    family: 'quadbin',
    resolutions: [26, 8],
    output: {
      levels: [withoutIntermediateColumns, last],
      levelCounts: createTransientView(graph, 'late-level-counts', 'uint32', 2)
    }
  });
  expectUnchangedGraphOnFailure(
    graph,
    () => pyramid.getCommandNodes(graph),
    /cell-pyramid input\/output field.*chunk.*limit/
  );
  device.destroy();
});

it('pyramid and rollup preflight global scratch and foreign later levels without mutations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [12_000_000, 12_000_000, 12_000_000]);
  const pyramid = new GPUCellPyramid({
    ...rows,
    scratchMode: 'packed',
    family: 'quadbin',
    resolutions: [26, 8],
    output: {levels: [createTable(graph), createTable(graph)]}
  });
  expectUnchangedGraphOnFailure(graph, () => pyramid.getCommandNodes(graph), /row scratch.*limit/);
  const other = new GPUCommandGraph(device);
  const small = createRows(graph);
  const foreign = new GPUCellPyramid({
    ...small,
    family: 'quadbin',
    resolutions: [26, 8],
    output: {levels: [createTable(graph), createTable(other)]}
  });
  expectUnchangedGraphOnFailure(graph, () => foreign.getCommandNodes(graph), /target graph/);
  const source = createTable(graph);
  const backing = createTransientView(graph, 'oversized-rollup-overflow', 'uint32', 33_554_433);
  source.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 33_554_432,
    byteOffset: 4
  });
  const rollup = new GPUCellRollup({
    family: 'quadbin',
    sourceResolution: 26,
    resolution: 8,
    source,
    output: createTable(graph)
  });
  expectUnchangedGraphOnFailure(
    graph,
    () => rollup.getCommandNodes(graph),
    /input\/output field.*chunk.*limit/
  );
  device.destroy();
});

it('chunk parameter and level-slot scratch checks precede every declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1, 1, 1]);
  const aggregation = new GPUCellAggregation({
    ...rows,
    family: 'quadbin',
    resolution: 8,
    output: createTable(graph, 4)
  });
  const pyramid = new GPUCellPyramid({
    positions: rows.positions.data[0],
    values: rows.values.data[0],
    family: 'quadbin',
    resolutions: [26, 8, 0],
    output: {
      levels: [createTable(graph, 4), createTable(graph, 4), createTable(graph, 4)],
      levelCounts: createTransientView(graph, 'slot-counts', 'uint32', 3)
    }
  });
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 512});
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 512, configurable: true});
  expectUnchangedGraphOnFailure(
    graph,
    () => aggregation.getCommandNodes(graph),
    /chunk parameter buffer byteLength.*512/
  );
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 268435456});
  expectUnchangedGraphOnFailure(
    graph,
    () => pyramid.getCommandNodes(graph),
    /level count slots scratch.*768.*limit 512/
  );
  device.destroy();
});

it('aggregation fields, extra counts and separate pyramid levels cannot overlap', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph);
  const table = createTable(graph);
  const props = {...rows, family: 'quadbin' as const, resolution: 26};
  expect(
    () => new GPUCellAggregation({...props, output: {...table, counts: table.status.count}})
  ).toThrow(/same length/);
  const overlappingCounts = graph.createDataView(table.sums.buffer, {
    format: 'uint32',
    length: table.cells.length
  });
  expect(
    () => new GPUCellAggregation({...props, output: {...table, counts: overlappingCounts}})
  ).toThrow(/must not overlap/);
  expect(() =>
    createCellAggregation({...props, output: table, extraCounts: [table.status.count]})
  ).toThrow(/must not overlap/);
  expect(
    () => new GPUCellPyramid({...props, resolutions: [26, 8], output: {levels: [table, table]}})
  ).toThrow(/must not overlap/);
  device.destroy();
});

it('point output binding prefixes are checked before creating any node or resource', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1]);
  const backing = createTransientView(graph, 'late-validity-backing', 'uint32', 64);
  const validity = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 252
  });
  const point = new GPUPointToCell({
    ...rows,
    family: 'quadbin',
    resolution: 8,
    output: {...rows.output, validity: createVectorView('late-validity', 'uint32', [validity])}
  });
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 128});
  expectUnchangedGraphOnFailure(
    graph,
    () => point.getCommandNodes(graph),
    /row chunk.*256.*limit 128/
  );
  device.destroy();
});

it('level selection rejects overlapping outputs and oversized aligned views without mutations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const levelCounts = createTransientView(graph, 'selection-level-counts', 'uint32', 3);
  const activeLevel = createTransientView(graph, 'selection-active-level', 'uint32', 1);
  const count = createTransientView(graph, 'selection-count', 'uint32', 1);
  expect(
    () =>
      new GPUCellLevelSelection({
        levelCounts,
        activeLevel,
        output: {status: {count}, firstRow: count}
      })
  ).toThrow(/must not overlap/);
  const backing = createTransientView(graph, 'selection-draw-backing', 'uint32', 67);
  const drawArguments = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 4,
    byteOffset: 252
  });
  const selection = new GPUCellLevelSelection({
    levelCounts,
    activeLevel,
    levelFirstRows: [8, 48, 128],
    output: {status: {count}, drawArguments}
  });
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 128});
  expectUnchangedGraphOnFailure(
    graph,
    () => selection.getCommandNodes(graph),
    /input\/output field.*chunk.*268.*limit 128/
  );
  device.destroy();
});

it('level selection source and pipelines stay constant as level count grows', () => {
  const sizes = [2, 100, 101].map(levelCount => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const selection = new GPUCellLevelSelection({
      levelCounts: createTransientView(graph, 'level-counts', 'uint32', levelCount),
      activeLevel: createTransientView(graph, 'active-level', 'uint32', 1),
      levelFirstRows: Array.from({length: levelCount}, (_, level) => level * 8),
      output: {status: {count: createTransientView(graph, 'count', 'uint32', 1)}}
    });
    const nodes = selection.getCommandNodes(graph);
    checkIdentifiers(nodes);
    const keys = getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    device.destroy();
    return keys;
  });
  expect([...sizes[0]]).toEqual([...sizes[1]]);
  expect([...sizes[1]]).toEqual([...sizes[2]]);
  expect(sizes[0].size).toBe(1);
});

it('packed point inputs preserve the public packed output type', () => {
  expectTypeOf<GPUPointToCellOutput['cells']>().toEqualTypeOf<GraphDataView<'uint32x2'>>();
  expectTypeOf<GPUPointToCellProps['output']['cells']>().toEqualTypeOf<GraphDataView<'uint32x2'>>();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1]);
  const packed = new GPUPointToCell({
    family: 'quadbin',
    resolution: 8,
    positions: rows.positions.data[0],
    output: {cells: rows.output.cells.data[0]}
  });
  expectTypeOf(packed.props.output.cells).toEqualTypeOf<GraphDataView<'uint32x2'>>();
  const chunked = new GPUPointToCell({...rows, family: 'quadbin', resolution: 8});
  expectTypeOf(chunked.props.output.cells).toEqualTypeOf<GraphVectorView<'uint32x2'>>();
  device.destroy();
});

it('uneven chunks reuse pipelines and dispatch their own workgroup counts', () => {
  const pipelineSets = [1, 2, 3].map(repetitions => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const lengths = Array.from({length: repetitions}, () => [1, 255, 256, 257, 513]).flat();
    const rows = createRows(graph, lengths, true);
    const contributor = new GPUCellAggregation({
      ...rows,
      family: 'quadbin',
      resolution: 26,
      output: createTable(graph)
    });
    const nodes = contributor
      .getCommandNodes(graph)
      .filter(node => /-(keys|values)-chunk-/.test(node.id));
    const keys = getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cacheShaders: false, _cachePipelines: false}
    });
    Object.defineProperty(device, 'createComputePipeline', {
      value: () => ({id: 'recording', hash: '', destroy() {}})
    });
    const dispatch = vi.fn();
    const buffers = new Map<GraphDataView['buffer'], ReturnType<typeof device.createBuffer>>();
    for (const node of nodes) {
      if (node.type !== 'compute') continue;
      const executable = node.compile({device});
      executable.encode({
        computePass: {setPipeline() {}, setBindings() {}, dispatch},
        getBuffer: (handle: GraphDataView['buffer']) => {
          let buffer = buffers.get(handle);
          if (!buffer) {
            buffer = device.createBuffer({byteLength: handle.byteLength, usage: handle.usage});
            buffers.set(handle, buffer);
          }
          return buffer;
        }
      } as never);
      executable.destroy?.();
    }
    const expectedGroups = lengths.flatMap(length => [
      [Math.ceil(length / 256), 1, 1],
      [Math.ceil(length / 256), 1, 1]
    ]);
    expect(dispatch.mock.calls).toEqual(expectedGroups);
    expect(dispatch.mock.calls).toHaveLength(expectedGroups.length);
    for (const buffer of buffers.values()) buffer.destroy();
    const point = new GPUPointToCell({...rows, family: 'quadbin', resolution: 8});
    expect(
      getGPUCommandNodePipelineKeys(point.getCommandNodes(graph), {includeShaderLayout: true}).size
    ).toBe(1);
    device.destroy();
    return keys;
  });
  expect(pipelineSets[0].size).toBe(2);
  expect([...pipelineSets[0]]).toEqual([...pipelineSets[1]]);
  expect([...pipelineSets[1]]).toEqual([...pipelineSets[2]]);
});

it('pyramid table kernels reuse sources and pipelines as level count grows', () => {
  const pipelineSets = [2, 3, 5].map(levelCount => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const rows = createRows(graph);
    const pyramid = new GPUCellPyramid({
      ...rows,
      family: 'quadbin',
      resolutions: Array.from({length: levelCount}, (_, index) => 12 - index),
      output: {
        levels: Array.from({length: levelCount}, () => createTable(graph)),
        levelCounts: createTransientView(graph, 'level-counts', 'uint32', levelCount)
      }
    });
    const nodes = pyramid.getCommandNodes(graph);
    checkIdentifiers(nodes);
    getParsedKernelSources(nodes);
    const keys = getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    device.destroy();
    return keys;
  });
  expect([...pipelineSets[0]].sort()).toEqual([...pipelineSets[1]].sort());
  expect([...pipelineSets[1]].sort()).toEqual([...pipelineSets[2]].sort());
});

it('adjacent logical output chunks can share an alignment window, co-bound fields cannot', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1, 1]);
  const shared = createTransientView(graph, 'adjacent-cells', 'uint32x2', 3);
  const cells = createVectorView(
    'adjacent-cells-chunks',
    'uint32x2',
    [0, 8].map(byteOffset =>
      graph.createDataView(shared.buffer, {format: 'uint32x2', length: 1, byteOffset})
    )
  );
  const props = {...rows, family: 'quadbin' as const, resolution: 8};
  expect(() => new GPUPointToCell({...props, output: {...rows.output, cells}})).not.toThrow();
  const validity = createVectorView(
    'adjacent-validity',
    'uint32',
    [16, 20].map(byteOffset =>
      graph.createDataView(shared.buffer, {format: 'uint32', length: 1, byteOffset})
    )
  );
  expect(() => new GPUPointToCell({...props, output: {cells, validity}})).toThrow(
    /aligned binding.*must not overlap/
  );
  device.destroy();
});

it('table parameter and publish limits fail before any graph declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1]);
  const output = createTable(graph, 1);
  const aggregation = new GPUCellAggregation({...rows, family: 'quadbin', resolution: 8, output});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
    value: 64,
    configurable: true
  });
  expectUnchangedGraphOnFailure(
    graph,
    () => aggregation.getCommandNodes(graph),
    /cell level parameters.*128.*limit 64/
  );
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 134217728});
  const tooMany = createCellAggregation({
    ...rows,
    family: 'quadbin',
    resolution: 8,
    output,
    extraCounts: Array.from({length: 5}, (_, index) =>
      createTransientView(graph, `extra-${index}`, 'uint32', 1)
    )
  });
  expectUnchangedGraphOnFailure(
    graph,
    () => tooMany.getCommandNodes(graph),
    /publish storage bindings.*8/
  );
  device.destroy();
});

it('scalar global scratch fits exactly 2^25 rows and rejects the next row without mutations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const exact = createRows(graph, [2 ** 24, 2 ** 24]);
  const above = createRows(graph, [2 ** 24, 2 ** 24, 1]);
  const contributor = new GPUCellAggregation({
    ...exact,
    family: 'quadbin',
    resolution: 26,
    output: createTable(graph)
  });
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  const tooLarge = new GPUCellAggregation({
    ...above,
    id: 'above',
    scratchMode: 'packed',
    family: 'quadbin',
    resolution: 26,
    output: createTable(graph)
  });
  expectUnchangedGraphOnFailure(
    graph,
    () => tooLarge.getCommandNodes(graph),
    /row scratch.*134217732.*limit 134217728/
  );
  const tiled = new GPUCellAggregation({
    ...above,
    id: 'tiled-above',
    family: 'quadbin',
    resolution: 26,
    output: createTable(graph)
  });
  expect(tiled.usesTiledScratch).toBe(true);
  expect(() => tiled.getCommandNodes(graph)).not.toThrow();
  device.destroy();
});

it('input buffers and derived scratch obey maxBufferSize before declaring resources', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1]);
  const backing = createTransientView(graph, 'large-position-buffer', 'float32x2', 256);
  const positions = createVectorView('small-position-subview', 'float32x2', [
    graph.createDataView(backing.buffer, {format: 'float32x2', length: 1})
  ]);
  const point = new GPUPointToCell({...rows, positions, family: 'quadbin', resolution: 8});
  const aggregation = new GPUCellAggregation({
    ...rows,
    positions,
    family: 'quadbin',
    resolution: 8,
    output: createTable(graph)
  });
  const smallChunks = createRows(graph, [128, 128, 1]);
  const scratch = new GPUCellAggregation({
    ...smallChunks,
    family: 'quadbin',
    resolution: 8,
    output: createTable(graph)
  });
  // Each binding is below 128 MiB; maxBufferSize independently bounds whole allocations.
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 1024});
  expectUnchangedGraphOnFailure(
    graph,
    () => point.getCommandNodes(graph),
    /row chunk buffer byteLength.*1024/
  );
  expectUnchangedGraphOnFailure(
    graph,
    () => aggregation.getCommandNodes(graph),
    /input\/output field 0 buffer byteLength.*1024/
  );
  expectUnchangedGraphOnFailure(
    graph,
    () => scratch.getCommandNodes(graph),
    /row scratch byteLength.*1024/
  );
  device.destroy();
});

it('different contributor IDs namespace chunk parameters and all derived resources', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const id of ['first', 'second']) {
    const rows = createRows(graph, [1, 255, 257]);
    expect(() =>
      graph.add(new GPUPointToCell({...rows, id: `point-${id}`, family: 'quadbin', resolution: 8}))
    ).not.toThrow();
    expect(() =>
      graph.add(
        new GPUCellAggregation({
          ...rows,
          id: `aggregation-${id}`,
          family: 'quadbin',
          resolution: 8,
          output: createTable(graph)
        })
      )
    ).not.toThrow();
    expect(() =>
      graph.add(
        new GPUCellPyramid({
          ...rows,
          id: `pyramid-${id}`,
          family: 'quadbin',
          resolutions: [12, 8, 0],
          output: {
            levels: [createTable(graph), createTable(graph), createTable(graph)],
            levelCounts: createTransientView(graph, `level-counts-${id}`, 'uint32', 3)
          }
        })
      )
    ).not.toThrow();
  }
  device.destroy();
});

it('invalid numeric shader props are refused at construction and float32 endpoints generate finite literals', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = createRows(graph, [1, 255, 257]);
  const output = createTable(graph);
  const props = {...rows, family: 'quadbin' as const, resolution: 8, output};
  const source = createTable(graph);
  for (const sumScale of [NaN, Infinity, -Infinity, Number.MAX_VALUE, 2 ** 128, 0, -1, 0.1]) {
    expect(() => new GPUCellAggregation({...props, sumScale})).toThrow(
      /cell-aggregation sumScale.*positive finite float32/
    );
    expect(
      () =>
        new GPUCellRollup({
          family: 'quadbin',
          sourceResolution: 8,
          resolution: 4,
          source,
          output,
          sumScale
        })
    ).toThrow(/cell-rollup sumScale.*positive finite float32/);
    expect(
      () =>
        new GPUCellPyramid({
          ...rows,
          family: 'quadbin',
          resolutions: [8, 4],
          sumScale,
          output: {levels: [source, output]}
        })
    ).toThrow(/sumScale.*positive finite float32/);
  }
  for (const resolution of [NaN, Infinity, -Infinity, Number.MAX_VALUE, 2 ** 32, 0.5]) {
    expect(() => new GPUCellAggregation({...props, resolution})).toThrow(/resolution/);
    expect(
      () => new GPUCellRollup({family: 'quadbin', sourceResolution: 8, resolution, source, output})
    ).toThrow(/resolution/);
    expect(
      () =>
        new GPUCellPyramid({
          ...rows,
          family: 'quadbin',
          resolutions: [resolution, 0],
          output: {levels: [source, output]}
        })
    ).toThrow(/resolution/);
    for (const family of ['quadbin', 'quadkey', 'geohash', 's2', 'h3'] as const) {
      expect(() => new GPUPointToCell({...rows, family, resolution})).toThrow(/resolution/);
    }
  }
  for (const [index, sumScale] of [2 ** -149, Math.fround(3.4028234663852886e38)].entries()) {
    const contributor = new GPUCellAggregation({...props, id: `finite-scale-${index}`, sumScale});
    const nodes = contributor.getCommandNodes(graph);
    checkIdentifiers(nodes);
    getParsedKernelSources(nodes);
    const sources = getGPUCommandNodeKernelSources(nodes);
    const literals = sources.flatMap(kernel => [
      ...kernel.source.matchAll(/const SUM_SCALE: f32 = ([^;]+);/g)
    ]);
    expect(literals.length).toBeGreaterThan(0);
    for (const literal of literals) {
      expect(Math.fround(Number(literal[1]))).toBe(sumScale);
      expect(Number.isFinite(Math.fround(Number(literal[1])))).toBe(true);
    }
  }
  device.destroy();
});
