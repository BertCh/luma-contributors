// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type VertexFormat} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline/gpu-cell-set-outline';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover/gpu-cell-cover';
import {GPUHilbertKeys} from '../../../src/gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys';
import {GPUHealpixCellAggregation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-cell-aggregation';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {GPUCellGridPath} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {createZonalStatisticsPrepareNodes} from '../../../src/gpu-spatial-analysis/zonal-statistics/zonal-statistics-passes';
import {
  createZonalStatisticsMeanNode,
  createZonalStatisticsDensityNode,
  createZonalStatisticsWeightedDensityNode,
  createZonalStatisticsVarianceNode
} from '../../../src/gpu-spatial-analysis/zonal-statistics/zonal-statistics-passes';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  WGSL_RESERVED_WORDS,
  auditWGSLModule
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectCellBarrierEntry, expectCellStorageScopes} from './cell-dispatch-audit';
import {coverS2OnCPU} from '../cell-cover/cell-cover-s2-oracle';
import {createRectangleRing} from '../cell-cover/cell-cover-oracle';

let sequence = 0;
function createFields(graph: GPUCommandGraph) {
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `review-${sequence++}`, format, length);
  const status = () => ({
    count: view('uint32', 1),
    overflow: view('uint32', 1),
    requiredCount: view('uint32', 1)
  });
  return {view, status};
}
function createCovering(graph: GPUCommandGraph, family: 'cap' | 'polygon', level: number) {
  const {view} = createFields(graph);
  const output = {
    cells: view('uint32x2', 16),
    counts: view('uint32x2', 1),
    status: view('uint32', 1)
  };
  return family === 'cap'
    ? new GPUS2CapCovering({
        caps: view('float32x4', 1),
        level,
        capacity: 16,
        frontierCapacity: 257,
        refinement: 'level-synchronous',
        output
      })
    : new GPUS2PolygonCovering({
        positions: view('float32x3', 3),
        ringOffsets: view('uint32', 2),
        maximumLevel: level,
        maximumVisitedCells: 257,
        capacity: 16,
        refinement: 'level-synchronous',
        output
      });
}

it('the independent S2 center-cover oracle identifies the face root and excludes a hole', () => {
  const shell = createRectangleRing([-20, -20, 20, 20]);
  const hole = createRectangleRing([-2, -2, 2, 2]);
  expect(coverS2OnCPU([[[shell]]], 0)).toEqual({cells: [1n << 60n], featureRows: [0]});
  expect(coverS2OnCPU([[[shell, hole]]], 0)).toEqual({cells: [], featureRows: []});
});

it('the cooperative entry audit distinguishes uniform group guards from divergent lane guards', () => {
  expect(() =>
    expectCellBarrierEntry(
      'fn main() { if (workgroupIndex >= BLOCK_COUNT) { return; } workgroupBarrier(); }'
    )
  ).not.toThrow();
  expect(() =>
    expectCellBarrierEntry(
      'fn main() { if (index >= ELEMENT_COUNT) { return; } workgroupBarrier(); }'
    )
  ).toThrow();
});

it.each([
  'cap',
  'polygon'
] as const)('S2 %s covering uses one classify/write pipeline for every runtime round', family => {
  const device = createNullWebGPUDevice();
  try {
    for (const level of [2, 5, 10]) {
      const graph = new GPUCommandGraph(device);
      const contributor = createCovering(graph, family, level);
      const nodes = contributor.getCommandNodes(graph);
      expectCellStorageScopes(nodes);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      for (const stage of ['classify', 'write']) {
        const selected = nodes.filter(node => node.id.includes(`-${stage}-`));
        expect(selected).toHaveLength(level + 1);
        expect(getGPUCommandNodePipelineKeys(selected).size).toBe(1);
        for (const kernel of getGPUCommandNodeKernelSources(selected)) {
          expect(kernel.constants).not.toHaveProperty('ROUND');
          expect(() => new WgslReflect(kernel.source)).not.toThrow();
          expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
          const words =
            kernel.source
              .replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '')
              .match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
          expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
        }
      }
    }
  } finally {
    device.destroy();
  }
});

it('S2 outline shares exact face neighbors and geometry, with valid reserved-word-free WGSL', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view, status} = createFields(graph);
    const contributor = new GPUCellSetOutline({
      family: 's2',
      cells: view('uint32x2', 24),
      output: {
        rows: view('uint32', 96),
        cells: view('uint32x2', 96),
        edgeIndices: view('uint32', 96),
        endpoints: view('float32x4', 96),
        status: status()
      }
    });
    expect(contributor.edgeSlots).toBe(4);
    expect(contributor.edgeModel).toBe('great-circle');
    expect(contributor.interiorSide).toBe('right');
    for (const {source, constants} of getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph)
    )) {
      expect(() => new WgslReflect(source)).not.toThrow();
      expectCellBarrierEntry(source);
      expect(auditWGSLModule(source, constants)).toEqual([]);
      const words =
        source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'quadbin',
  's2'
] as const)('%s grid path keeps one shader and per-chunk workgroups for unequal subviews', family => {
  const device = createNullWebGPUDevice();
  try {
    const counts: number[] = [];
    for (const chunkCount of [1, 3, 9]) {
      const graph = new GPUCommandGraph(device);
      const {view} = createFields(graph);
      const lengths = Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]);
      const field = <Format extends 'uint32' | 'uint32x2'>(
        name: string,
        format: Format,
        stride = 1
      ) =>
        createVectorView(
          name,
          format,
          lengths.map((length, index) => {
            const backing = view('uint32', length * stride * (format === 'uint32x2' ? 2 : 1) + 65);
            return graph.createDataView(backing.buffer, {
              format,
              length: length * stride,
              byteOffset: [4, 12, 260][index % 3]
            });
          })
        );
      const contributor = new GPUCellGridPath({
        family,
        origins: field('origin', 'uint32x2'),
        destinations: field('destination', 'uint32x2'),
        mask: field('mask', 'uint32'),
        maximumPathLength: 4,
        output: {
          cells: field('path', 'uint32x2', 4),
          distances: field('distance', 'uint32'),
          counts: field('count', 'uint32')
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      expectCellStorageScopes(nodes);
      expect(nodes.slice(1).map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        lengths.map(length => Math.ceil(length / 256))
      );
      counts.push(getGPUCommandNodePipelineKeys(nodes).size);
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(source)).not.toThrow();
        expectCellBarrierEntry(source);
        expect(auditWGSLModule(source, constants)).toEqual([]);
        const words =
          source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
    }
    expect(counts).toEqual([1, 1, 1]);
  } finally {
    device.destroy();
  }
});

it('zonal preparation shares one shader and dispatches every chunk at its own length', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const chunkCount of [1, 3, 9]) {
      const graph = new GPUCommandGraph(device);
      const {view} = createFields(graph);
      const lengths = Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]);
      const field = <Format extends 'uint32' | 'float32'>(name: string, format: Format) =>
        createVectorView(
          name,
          format,
          lengths.map((length, index) => {
            const backing = view(format, length + 65);
            return graph.createDataView(backing.buffer, {
              format,
              length,
              byteOffset: [4, 12, 260][index % 3]
            });
          })
        );
      const nodes = createZonalStatisticsPrepareNodes(graph, {
        id: 'zonal-review',
        values: field('values', 'float32'),
        weights: field('weights', 'float32'),
        valueMask: field('mask', 'uint32'),
        sumContributions: field('sum', 'float32'),
        weightContributions: field('weight-sum', 'float32')
      });
      expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(1);
      expect(nodes.slice(1).map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        lengths.map(length => Math.ceil(length / 256))
      );
    }
  } finally {
    device.destroy();
  }
});

it('Hilbert sort accepts disjoint packed output bytes despite the published sort buffer restriction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view} = createFields(graph);
    const backing = view('uint32', 20);
    const field = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'uint32', length: 4, byteOffset});
    const contributor = new GPUHilbertKeys({
      positions: view('float32x2', 4),
      bounds: view('float32', 4),
      output: {keys: field(4), sortedRows: field(20), sortedKeys: field(36)}
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('HEALPix aggregation validates mask/value formats, scalar status and disjoint separate reduction outputs', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view, status} = createFields(graph);
    const backing = view('float32', 12);
    const reduction = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'float32', length: 2, byteOffset});
    const props = {
      order: 2,
      cellCapacity: 2,
      cells: view('uint32x2', 4),
      values: view('float32', 4),
      output: {
        cells: view('uint32x2', 2),
        counts: view('uint32', 2),
        sums: reduction(4),
        minimums: reduction(12),
        maximums: reduction(20),
        status: status()
      }
    };
    expect(
      () => new GPUHealpixCellAggregation({...props, mask: view('float32', 4) as never})
    ).toThrow(/mask.*uint32/);
    expect(
      () => new GPUHealpixCellAggregation({...props, values: view('uint32', 4) as never})
    ).toThrow(/values.*float32/);
    expect(
      () =>
        new GPUHealpixCellAggregation({
          ...props,
          output: {...props.output, status: {...props.output.status, count: view('uint32', 0)}}
        })
    ).toThrow(/count.*one row/);
    expect(() => new GPUHealpixCellAggregation(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('new preflights reject STORAGE omissions and nested scratch before any graph mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view, status} = createFields(graph);
    const noStorage = graph.createTransientBuffer({
      id: 'not-storage',
      usage: Buffer.COPY_DST,
      byteLength: 16
    });
    const contributor = new GPUHilbertKeys({
      positions: graph.createDataView(noStorage, {format: 'float32x2', length: 2}),
      output: {keys: view('uint32', 2)}
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/usage.*STORAGE/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    const cover = new GPUCellCover({
      family: 'quadbin',
      resolution: 3,
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2),
      candidateCapacity: 8,
      candidateEnumeration: 'scanline',
      scanlineSlotCapacity: 8,
      output: {
        featureIds: view('uint32', 2),
        cells: view('uint32x2', 2),
        status: {...status(), candidateOverflow: view('uint32', 1)}
      }
    });
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 128});
    const allocateCover = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => cover.getCommandNodes(graph)).toThrow(/range scratch|scanline/);
    expect(allocateCover).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 512});
    const zonal = new GPUZonalStatistics({
      positions: view('float32x2', 10),
      values: view('float32', 10),
      features: {
        kind: 'polygons',
        positions: view('float32x2', 20),
        featureOffsets: view('uint32', 5),
        polygonOffsets: view('uint32', 5),
        ringOffsets: view('uint32', 6),
        candidateCapacity: 16,
        leafCapacity: 128
      },
      output: {counts: view('uint32', 4)}
    });
    const allocateZonal = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => zonal.getCommandNodes(graph)).toThrow(/join BVH node bounds/);
    expect(allocateZonal).not.toHaveBeenCalled();
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

it('S2 center covering has fixed pipelines across levels, bounded private memory and named limits', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view, status} = createFields(graph);
    const separateStatus = view('uint32', 3);
    const props = {
      family: 's2' as const,
      resolution: 5,
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2),
      candidateCapacity: 257,
      featureIds: view('uint32', 1),
      output: {
        featureIds: view('uint32', 16),
        cells: view('uint32x2', 16),
        core: view('uint32', 16),
        status: {
          ...status(),
          candidateOverflow: view('uint32', 1),
          candidateCount: view('uint32', 1),
          // These are cleared in separate dispatches, so sharing the aligned binding prefix
          // is valid as long as their logical bytes remain disjoint.
          invalidCount: graph.createDataView(separateStatus.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 4
          }),
          uncertainCount: graph.createDataView(separateStatus.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 8
          })
        }
      }
    };
    expect(() => new GPUCellCover({...props, resolution: 16})).toThrow(/s2 resolution/);
    expect(() => new GPUCellCover({...props, containment: 'intersects'})).toThrow(
      /s2 supports only/
    );
    for (const resolution of [0, 3, 8, 15]) {
      const contributor = new GPUCellCover({...props, id: `s2-cover-${resolution}`, resolution});
      const nodes = contributor.getCommandNodes(graph);
      expectCellStorageScopes(nodes);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount!);
      for (const stage of ['classify', 'write']) {
        const selected = nodes.filter(node => node.id.includes(`-s2-${stage}-`));
        expect(getGPUCommandNodePipelineKeys(selected).size).toBe(
          stage === 'write' && resolution === 0 ? 0 : 1
        );
      }
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(source)).not.toThrow();
        expectCellBarrierEntry(source);
        expect(auditWGSLModule(source, constants)).toEqual([]);
        const words =
          source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('construction bounds new count literals and rejects inherited geometry families', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view, status} = createFields(graph);
    expect(
      () =>
        new GPUHilbertKeys({
          positions: view('float32x2', 2 ** 31),
          output: {keys: view('uint32', 2 ** 31)}
        })
    ).toThrow(/itemCount/);
    const aggregation = {
      order: 2,
      cellCapacity: 1,
      cells: view('uint32x2', 4),
      output: {cells: view('uint32x2', 1), counts: view('uint32', 1), status: status()}
    };
    expect(() => new GPUHealpixCellAggregation({...aggregation, scheme: 'bad' as never})).toThrow(
      /scheme/
    );
    expect(
      () => new GPUHealpixCellAggregation({...aggregation, cells: view('uint32x2', 2 ** 31)})
    ).toThrow(/cells.length/);
    for (const family of ['toString', 'constructor', '__proto__'])
      expect(
        () =>
          new GPUCellGeometry({
            id: 'geometry-family',
            family: family as never,
            cells: view('uint32x2', 1),
            output: {centers: view('float32x2', 1)}
          })
      ).toThrow(/^geometry-family family/);
    for (const maximumSearchRadius of [NaN, Infinity, 0xffffffff, -1, 1.5])
      expect(
        () =>
          new GPUCellGridPath({
            family: 's2',
            maximumSearchRadius,
            origins: view('uint32x2', 1),
            destinations: view('uint32x2', 1),
            output: {distances: view('uint32', 1)}
          })
      ).toThrow(/maximumSearchRadius/);
  } finally {
    device.destroy();
  }
});

it('zonal count-only aggregation ignores unused value and area storage usage', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view} = createFields(graph);
    const unused = graph.createTransientBuffer({
      id: 'unused-values-and-areas',
      byteLength: 64,
      usage: Buffer.COPY_DST
    });
    const values = graph.createDataView(unused, {format: 'float32', length: 4});
    const areas = graph.createDataView(unused, {format: 'float32', length: 2, byteOffset: 32});
    const contributor = new GPUZonalStatistics({
      features: {kind: 'feature-rows', pointFeatureRows: view('uint32', 4), featureCount: 2},
      values,
      areas,
      output: {counts: view('uint32', 2)}
    });
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('zonal construction rejects mixed physical storage scopes and permits independent packed reductions', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view} = createFields(graph);
    const packed = view('float32', 132);
    const first = graph.createDataView(packed.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 4
    });
    const second = graph.createDataView(packed.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 260
    });
    const base = {
      values: view('float32', 4),
      features: {
        kind: 'feature-rows' as const,
        featureCount: 2,
        pointFeatureRows: view('uint32', 4)
      }
    };
    expect(() => new GPUZonalStatistics({...base, output: {sums: first, means: second}})).toThrow(
      /physical buffer.*read-only and writable/
    );
    const counts = graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 2,
      byteOffset: 260
    });
    const nodes = new GPUZonalStatistics({...base, output: {sums: first, counts}}).getCommandNodes(
      graph
    );
    expectCellStorageScopes(nodes);
  } finally {
    device.destroy();
  }
});

it('Hilbert and zonal finite quotients use bounded significands on Metal', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const {view} = createFields(graph);
    const hilbert = new GPUHilbertKeys({
      positions: view('float32x2', 257),
      output: {keys: view('uint32', 257)}
    }).getCommandNodes(graph);
    const float = () => view('float32', 2);
    const count = () => view('uint32', 2);
    const zonal = [
      createZonalStatisticsMeanNode(graph, {
        id: 'mean',
        featureCount: 2,
        sums: float(),
        weightSums: float(),
        means: float()
      }),
      createZonalStatisticsDensityNode(graph, {
        id: 'density',
        featureCount: 2,
        counts: count(),
        areas: float(),
        densities: float()
      }),
      createZonalStatisticsWeightedDensityNode(graph, {
        id: 'weighted-density',
        featureCount: 2,
        weightSums: float(),
        areas: float(),
        weightedDensities: float()
      }),
      createZonalStatisticsVarianceNode(graph, {
        id: 'variance',
        featureCount: 2,
        valueCounts: count(),
        squaredDeviationSums: float(),
        weightSums: float(),
        weightSquareSums: float(),
        variances: float()
      })
    ];
    for (const {source, constants} of getGPUCommandNodeKernelSources([...hilbert, ...zonal])) {
      expectCellBarrierEntry(source);
      expect(auditWGSLModule(source, constants)).toEqual([]);
      expect(() => new WgslReflect(source)).not.toThrow();
      if (source.includes('fn quantizeAxis(')) {
        expect(source).toContain('getScaledQuotient(position - lower, upper - lower');
        expect(source).not.toContain('(value - minimum) / extent');
      } else if (source.includes('fn getZonalQuotient(')) {
        expect(source).toContain('getScaledQuotient(numerator, denominator, 1.0, 0)');
      }
    }
    expectCellStorageScopes([...hilbert, ...zonal]);
  } finally {
    device.destroy();
  }
});
