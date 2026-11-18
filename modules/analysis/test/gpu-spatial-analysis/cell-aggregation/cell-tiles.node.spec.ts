// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUCellAggregation,
  createCellAggregation
} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments,
  findConstantNonFiniteBitcasts
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function fixture(rows: number, tileLength = 16) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rowLengths = [3, 0, rows - 3];
  const positions = createVectorView(
    'positions',
    'float32x2',
    rowLengths.map((length, index) =>
      createTransientView(graph, `positions-${index}`, 'float32x2', length)
    )
  );
  const values = createVectorView(
    'values',
    'float32',
    rowLengths.map((length, index) =>
      createTransientView(graph, `values-${index}`, 'float32', length)
    )
  );
  const output = {
    cells: createTransientView(graph, 'cells', 'uint32x2', 4),
    counts: createTransientView(graph, 'counts', 'uint32', 4),
    sums: createTransientView(graph, 'sums', 'uint32x2', 4),
    sumValues: createTransientView(graph, 'sum-values', 'float32', 4),
    minimums: createTransientView(graph, 'minimums', 'float32', 4),
    maximums: createTransientView(graph, 'maximums', 'float32', 4),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1),
      requiredCount: createTransientView(graph, 'required', 'uint32', 1)
    }
  };
  return {
    device,
    graph,
    props: {
      positions,
      values,
      family: 'quadbin' as const,
      resolution: 26,
      scratchTileSize: tileLength,
      output
    }
  };
}

it('fixed cell tiles share every shader and pipeline as tile count and tails grow', () => {
  const sets = [49, 113, 225].map(rows => {
    const {device, graph, props} = fixture(rows);
    try {
      const contributor = new GPUCellAggregation(props);
      const nodes = contributor.getCommandNodes(graph);
      expect(contributor.usesTiledScratch).toBe(true);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      getParsedKernelSources(nodes);
      for (const {source, id} of getGPUCommandNodeKernelSources(nodes)) {
        expect(findConstantNonFiniteBitcasts(stripWGSLComments(source)), id).toEqual([]);
        expect(
          [...new Set(stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].filter(
            word => WGSL_RESERVED_WORDS.has(word)
          ),
          id
        ).toEqual([]);
      }
      return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
    } finally {
      device.destroy();
    }
  });
  expect([...sets[0]].sort()).toEqual([...sets[1]].sort());
  expect([...sets[0]].sort()).toEqual([...sets[2]].sort());
});

it('tiled extrema build empty-row NaNs at runtime using the shared WGSL helper', () => {
  const {device, graph, props} = fixture(49);
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUCellAggregation(props).getCommandNodes(graph)
    );
    const finishes = sources.filter(({id}) => /-(minimums|maximums)-finish$/.test(id));
    expect(finishes).toHaveLength(2);
    for (const {source, id} of finishes) {
      expect(source, id).toContain('select(getNaN(), decodeOrderedKey(');
      expect(source, id).toContain('var bits = 0x7fc00000u;');
      expect(source, id).toContain('return bitcast<f32>(bits);');
      expect(findConstantNonFiniteBitcasts(stripWGSLComments(source)), id).toEqual([]);
    }
    // A parser accepts this expression; the semantic audit must catch it before WebGPU does.
    expect(findConstantNonFiniteBitcasts('let missing = bitcast<f32>(0x7fc00000u);')).toHaveLength(
      1
    );
  } finally {
    device.destroy();
  }
});

it('cell tiles preflight widest tile scratch before any allocation', () => {
  const {device, graph, props} = fixture(65, 64);
  try {
    // Inputs fit; the 64-bit local table needs 512 bytes.
    device.limits.maxStorageBufferBindingSize = 500;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => new GPUCellAggregation(props).getCommandNodes(graph)).toThrow();
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('tiled cell columns sharing one allocation never mix read-only and writable storage usage', () => {
  const {device, graph, props} = fixture(49);
  try {
    const buffer = graph.createTransientBuffer({
      id: 'shared-columns',
      byteLength: 9 * 256,
      usage: 128
    });
    const output = {
      cells: graph.createDataView(buffer, {format: 'uint32x2', length: 4, byteOffset: 0}),
      counts: graph.createDataView(buffer, {format: 'uint32', length: 4, byteOffset: 256}),
      sums: graph.createDataView(buffer, {format: 'uint32x2', length: 4, byteOffset: 512}),
      sumValues: graph.createDataView(buffer, {format: 'float32', length: 4, byteOffset: 768}),
      minimums: graph.createDataView(buffer, {format: 'float32', length: 4, byteOffset: 1024}),
      maximums: graph.createDataView(buffer, {format: 'float32', length: 4, byteOffset: 1280}),
      status: {
        count: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 1536}),
        overflow: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 1792}),
        requiredCount: graph.createDataView(buffer, {format: 'uint32', length: 1, byteOffset: 2048})
      }
    };
    // Columns publish in different dispatches; aligned prefixes may overlap logically
    // disjoint caller bytes. Status fields still need separate aligned bindings.
    const adjacentCounts = graph.createDataView(buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 32
    });
    expect(() =>
      new GPUCellAggregation({
        ...props,
        id: 'adjacent-columns',
        output: {...output, counts: adjacentCounts}
      }).getCommandNodes(graph)
    ).not.toThrow();
    expect(
      () =>
        new GPUCellAggregation({
          ...props,
          output: {
            ...output,
            status: {
              ...output.status,
              overflow: graph.createDataView(buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: 1540
              })
            }
          }
        })
    ).toThrow(/aligned binding/);
    for (const node of new GPUCellAggregation({...props, output}).getCommandNodes(graph)) {
      const usages = new Map<unknown, Set<string>>();
      for (const resource of node.resources ?? []) {
        if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) continue;
        const handle = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
        const set = usages.get(handle) ?? new Set<string>();
        set.add(resource.usage);
        usages.set(handle, set);
      }
      for (const set of usages.values())
        expect(set.has('storage-read') && set.has('storage-read-write'), node.id).toBe(false);
    }
    const overlap = graph.createDataView(buffer, {format: 'uint32', length: 4, byteOffset: 4});
    expect(() => new GPUCellAggregation({...props, output: {...output, counts: overlap}})).toThrow(
      /overlap/
    );
    expect(() =>
      createCellAggregation({...props, output, extraCounts: [output.status.count]})
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

it('cell tile preflight rejects missing STORAGE usage before declaring scratch', () => {
  const {device, graph, props} = fixture(49);
  try {
    const buffer = graph.createTransientBuffer({
      id: 'non-storage-output',
      usage: 8,
      byteLength: 16
    });
    const counts = graph.createDataView(buffer, {format: 'uint32', length: 4});
    const contributor = new GPUCellAggregation({...props, output: {...props.output, counts}});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/Buffer.STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('cell tile widths reject non-finite, unaddressable and non-power-of-two sizes at construction', () => {
  const {device, props} = fixture(49);
  try {
    for (const scratchTileSize of [NaN, Infinity, -1, 0, 3, 2 ** 31])
      expect(() => new GPUCellAggregation({...props, scratchTileSize})).toThrow(/scratchTileSize/);
  } finally {
    device.destroy();
  }
});
