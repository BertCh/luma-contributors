// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  createTransientView,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPULocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPULocalPermutationTest,
  GPUGlobalPermutationTest
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  GPURasterFalseDiscoveryRateMask,
  GPUMultipleTestingAdjustment
} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function createScene(rows: number, chunkSizes: readonly number[] = [rows]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `scale-${serial++}`, format, length);
  let rowStart = 0;
  const chunks = chunkSizes.map(rowCount => {
    const chunk = {
      rowStart,
      rowCount,
      weights: {
        offsets: view('uint32', rowCount + 1),
        neighbors: view('uint32', rowCount),
        weights: view('float32', rowCount)
      }
    };
    rowStart += rowCount;
    return chunk;
  });
  return {device, graph, view, weights: {rowCount: rows, chunks}, values: view('float32', rows)};
}

for (const Contributor of [GPUHotSpotAnalysis, GPULocalMoran]) {
  describe(Contributor.name, () => {
    it('declares 30 million rows, chunked CSR and named statuses within default binding limits', () => {
      const scene = createScene(30_000_000, [7_000_000, 8_000_000, 15_000_000]);
      try {
        const {graph, view, weights, values} = scene;
        const nodes = new Contributor({
          values,
          weights,
          parameters: view('float32', 8),
          output: {
            zScores: view('float32', values.length),
            neighborCounts: view('uint32', values.length),
            status: {invalidCount: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }).getCommandNodes(graph);
        expect(nodes.length).toBeGreaterThan(0);
        expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expect(findReservedIdentifiers(stripWGSLComments(kernel.source)), kernel.id).toEqual([]);
          expect(stripWGSLComments(kernel.source)).not.toMatch(/\b(?:nan|inf)\b/);
        }
      } finally {
        scene.device.destroy();
      }
    });

    it('keeps one pipeline per stage as unequal chunk counts grow, with each chunk dispatched at its own size', () => {
      const build = (sizes: number[]) => {
        const {device, graph, view, weights, values} = createScene(1024, sizes);
        const nodes = new Contributor({
          values,
          weights,
          parameters: view('float32', 8),
          output: {zScores: view('float32', 1024)}
        }).getCommandNodes(graph);
        const rowNodes = nodes.filter(node => /-(?:neighbors|row-states)-chunk/.test(node.id));
        const keys = getGPUCommandNodePipelineKeys(rowNodes);
        const work = rowNodes.map(
          node =>
            (node as {workload?: {maximumInvocationCount?: number}}).workload
              ?.maximumInvocationCount
        );
        device.destroy();
        return {keys, work};
      };
      expect(build([512, 512]).keys.size).toBe(2);
      const many = build([1, 31, 256, 736]);
      expect(many.keys.size).toBe(2);
      expect(many.work).toEqual([1, 31, 256, 736, 1, 31, 256, 736]);
    });

    it('rejects overlapping aligned writable ranges at construction', () => {
      const {device, graph, view, values, weights} = createScene(4);
      try {
        const storage = view('float32', 8);
        const zScores = graph.createDataView(storage.buffer, {format: 'float32', length: 4});
        const pValues = graph.createDataView(storage.buffer, {
          format: 'float32',
          length: 4,
          byteOffset: 16
        });
        expect(
          () =>
            new Contributor({
              values,
              weights,
              parameters: view('float32', 8),
              output: {zScores, pValues}
            })
        ).toThrow(/aligned.*overlap/);
      } finally {
        device.destroy();
      }
    });

    it('preflights aligned inputs and derived scratch before declaring graph resources', () => {
      for (const failure of ['aligned-input', 'scratch'] as const) {
        const {device, graph, view, weights, values} = createScene(8);
        try {
          const storage = view('float32', 65);
          const oversized = graph.createDataView(storage.buffer, {
            format: 'float32',
            length: 8,
            byteOffset: 4
          });
          Object.assign(device.limits, {
            maxStorageBufferBindingSize: failure === 'scratch' ? 52 : 32
          });
          const contributor = new Contributor({
            values: failure === 'scratch' ? values : oversized,
            weights,
            parameters: view('float32', 8),
            output: {zScores: view('float32', 8)}
          });
          const allocate = vi.spyOn(graph, 'createTransientBuffer');
          expect(() => contributor.getCommandNodes(graph)).toThrow(/binding|scratch/);
          expect(allocate).not.toHaveBeenCalled();
        } finally {
          device.destroy();
        }
      }
    });
  });
}

it('local and global permutation preparation supports 30 million exact uint32 row positions', () => {
  for (const local of [false, true]) {
    const {device, graph, view, weights, values} = createScene(30_000_000);
    try {
      const common = {values, weights, parameters: view('uint32', 4), maximumPermutations: 3};
      const contributor = local
        ? new GPULocalPermutationTest({
            ...common,
            statistic: 'localMoran',
            output: {
              exceedances: view('uint32', values.length),
              pseudoPValues: view('float32', values.length),
              status: {overflow: view('uint32', 1)}
            }
          })
        : new GPUGlobalPermutationTest({
            ...common,
            statistic: 'moran',
            output: {summary: view('float32', 12)}
          });
      const nodes = contributor.getCommandNodes(graph);
      if (!local)
        expect(nodes.length).toBeLessThanOrEqual(
          (contributor as GPUGlobalPermutationTest).estimatedCommandNodeCount
        );
      for (const node of nodes) {
        for (const resource of (
          node as {resources?: {buffer: GraphDataView | {byteLength: number}}[]}
        ).resources ?? []) {
          expect(
            'rowByteLength' in resource.buffer
              ? getViewBindingRange(resource.buffer).size
              : resource.buffer.byteLength
          ).toBeLessThanOrEqual(device.limits.maxStorageBufferBindingSize);
        }
      }
      expect(
        getParsedKernelSources(
          nodes.filter(
            node =>
              node.id.includes('-inputs-') ||
              (local && /-(?:observed|permute)(?:-|$)/.test(node.id))
          )
        ).size
      ).toBeGreaterThan(0);
      const sources = getGPUCommandNodeKernelSources(nodes);
      expect(sources.some(kernel => /rowPositions: array<u32>/.test(kernel.source))).toBe(true);
      for (const kernel of sources)
        expect(findReservedIdentifiers(stripWGSLComments(kernel.source)), kernel.id).toEqual([]);
    } finally {
      device.destroy();
    }
  }
});

it('permutation chunks and row tiles keep a constant number of pipelines', () => {
  const build = (sizes: number[], rowTileSize: number) => {
    const {device, graph, view, weights, values} = createScene(1024, sizes);
    const nodes = new GPULocalPermutationTest({
      values,
      weights,
      rowTileSize,
      parameters: view('uint32', 4),
      statistic: 'localMoran',
      maximumPermutations: 3,
      output: {
        exceedances: view('uint32', 1024),
        pseudoPValues: view('float32', 1024),
        status: {overflow: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const keys = getGPUCommandNodePipelineKeys(
      nodes.filter(node => /-(?:analysis|row-weights|observed|permute)(?:-|$)/.test(node.id))
    );
    device.destroy();
    return keys.size;
  };
  expect(build([512, 512], 512)).toBe(4);
  expect(build([1, 31, 256, 736], 17)).toBe(4);
});

it('multiple-testing corrections accept 30 million rows with uint32 ranks and family counts', () => {
  for (const method of [
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const {device, graph, view} = createScene(30_000_000);
    try {
      expect(() =>
        new GPUMultipleTestingAdjustment({
          pValues: view('float32', 30_000_000),
          method,
          output: {adjustedPValues: view('float32', 30_000_000), testedCount: view('uint32', 1)}
        }).getCommandNodes(graph)
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  }
});

it('local permutation row tiles validate their count and construction-time command budget', () => {
  for (const rowTileSize of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    const {device, view, weights, values} = createScene(4);
    try {
      expect(
        () =>
          new GPULocalPermutationTest({
            values,
            weights,
            rowTileSize,
            parameters: view('uint32', 4),
            statistic: 'localMoran',
            maximumPermutations: 3,
            output: {
              exceedances: view('uint32', 4),
              pseudoPValues: view('float32', 4),
              status: {overflow: view('uint32', 1)}
            }
          })
      ).toThrow(/rowTileSize/);
    } finally {
      device.destroy();
    }
  }
  const {device, view, weights, values} = createScene(100_000);
  try {
    expect(
      () =>
        new GPULocalPermutationTest({
          values,
          weights,
          rowTileSize: 1,
          parameters: view('uint32', 4),
          statistic: 'localMoran',
          maximumPermutations: 3,
          output: {
            exceedances: view('uint32', values.length),
            pseudoPValues: view('float32', values.length),
            status: {overflow: view('uint32', 1)}
          }
        })
    ).toThrow(/estimated command node count/);
  } finally {
    device.destroy();
  }
});

it('local Geary reuses two row-stage pipelines for unequal CSR chunks', () => {
  const {device, graph, view, weights, values} = createScene(1024, [1, 31, 256, 736]);
  try {
    const nodes = new GPULocalGeary({
      values,
      weights,
      parameters: view('float32', 8),
      output: {statistic: view('float32', 1024)}
    }).getCommandNodes(graph);
    expect(
      getGPUCommandNodePipelineKeys(
        nodes.filter(node => /-(?:row-states|statistic)-chunk/.test(node.id))
      ).size
    ).toBe(2);
  } finally {
    device.destroy();
  }
});

it('permutation contributors reject oversized aligned input bindings before any allocation', () => {
  for (const local of [false, true]) {
    const {device, graph, view, weights} = createScene(4);
    try {
      const storage = view('float32', 20);
      const values = graph.createDataView(storage.buffer, {
        format: 'float32',
        length: 4,
        byteOffset: 64
      });
      const common = {values, weights, parameters: view('uint32', 4), maximumPermutations: 3};
      const contributor = local
        ? new GPULocalPermutationTest({
            ...common,
            statistic: 'localMoran',
            output: {
              exceedances: view('uint32', 4),
              pseudoPValues: view('float32', 4),
              status: {overflow: view('uint32', 1)}
            }
          })
        : new GPUGlobalPermutationTest({
            ...common,
            statistic: 'moran',
            output: {summary: view('float32', 12)}
          });
      Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/80.*64/);
      expect(allocate).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('multiple testing rejects oversized output bindings before its sort and scan allocation', () => {
  const {device, graph, view} = createScene(4);
  try {
    const storage = view('float32', 20);
    const adjustedPValues = graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 4,
      byteOffset: 64
    });
    const contributor = new GPUMultipleTestingAdjustment({
      method: 'holm',
      pValues: view('float32', 4),
      output: {adjustedPValues}
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/80.*64/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('raster multiple-testing preflights aligned outputs before row scratch is allocated', () => {
  const {device, graph, view} = createScene(4);
  try {
    const storage = view('uint32', 20);
    const significant = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 4,
      byteOffset: 64
    });
    const contributor = new GPURasterFalseDiscoveryRateMask({
      cellCount: 4,
      pValues: view('float32', 4),
      parameters: view('float32', 1),
      output: {significant}
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 64});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/80.*64/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('local permutation command estimates cover tiled preparation, correction and optional outputs', () => {
  const {device, graph, view, weights, values} = createScene(1024, [1, 31, 256, 736]);
  try {
    const contributor = new GPULocalPermutationTest({
      values,
      weights,
      parameters: view('uint32', 4),
      statistic: 'localGeary',
      maximumPermutations: 999,
      rowTileSize: 17,
      multipleTesting: 'benjamini-yekutieli',
      output: {
        exceedances: view('uint32', 1024),
        pseudoPValues: view('float32', 1024),
        observed: view('float32', 1024),
        adjustedPValues: view('float32', 1024),
        classes: view('uint32', 1024),
        status: {overflow: view('uint32', 1), invalidCount: view('uint32', 1)}
      }
    });
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  } finally {
    device.destroy();
  }
});

it('raster multiple-testing validates minimumFamilySize at construction', () => {
  const {device, view} = createScene(4);
  try {
    const props = {
      cellCount: 4,
      pValues: view('float32', 4),
      parameters: view('float32', 1),
      output: {significant: view('uint32', 4)}
    };
    for (const minimumFamilySize of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, 2 ** 31 + 1]) {
      expect(() => new GPURasterFalseDiscoveryRateMask({...props, minimumFamilySize})).toThrow(
        /minimumFamilySize/
      );
    }
  } finally {
    device.destroy();
  }
});
