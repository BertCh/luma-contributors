// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {Device} from '@luma.gl/core';
import {getPolygonEdgeSlabNodes} from '../../src/gpu-spatial-analysis/spatial-join/polygon-edge-slabs';
import {CELLS_AND_JOINS_CASES} from '../analysis-core/kernel-audit/cases-cells-and-joins';
import {COVERAGE_GEOMETRY_CASES} from '../analysis-core/kernel-audit/cases-coverage-geometry';
import {GEOMETRY_CASES} from '../analysis-core/kernel-audit/cases-geometry';
import {NETWORK_CASES} from '../analysis-core/kernel-audit/cases-network';
import {RASTER_CASES} from '../analysis-core/kernel-audit/cases-raster';
import {STATISTICS_CASES} from '../analysis-core/kernel-audit/cases-statistics';
import {
  createKernelAuditContext,
  getDeclaredNodes
} from '../analysis-core/kernel-audit/kernel-audit-context';
import {
  getParsedKernelSources,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from './gpu-contributor-test-utils';

const CASES = [
  ...CELLS_AND_JOINS_CASES,
  ...COVERAGE_GEOMETRY_CASES,
  ...GEOMETRY_CASES,
  ...NETWORK_CASES,
  ...RASTER_CASES,
  ...STATISTICS_CASES
];

function requireStorageBudget(nodes: readonly unknown[], parseSources = true) {
  // Keep the existing raw-source sites independent of WGSL parser support.
  const configurations = parseSources ? undefined : getGPUCommandNodeKernelSources(nodes);
  const sources = configurations
    ? Object.assign(new Map(configurations.map(kernel => [kernel.id, kernel.source])), {
        configurations
      })
    : getParsedKernelSources(nodes);
  for (const {id, source} of sources.configurations)
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  return sources;
}

it.each([
  [true, false],
  [true, true],
  [false, false],
  [false, true]
] as const)('final3 H5 scan-total gate rejects nine/one shared-ID requests, parsed=%s reversed=%s', (parseSources, reversed) => {
  const counts = reversed ? [1, 9] : [9, 1];
  const requests = counts.map(count => ({
    compile: ({device}: {device: Device}) => {
      const source = Array.from(
        {length: count},
        (_, binding) =>
          `@group(0) @binding(${binding}) var<storage, read> data${binding}: array<u32>;`
      ).join('\n');
      const shader = device.createShader({stage: 'compute', source});
      device.createComputePipeline({id: 'shared', shader});
      return {destroy: () => shader.destroy()};
    }
  }));
  const sources = getParsedKernelSources(requests);
  expect(sources.configurations.map(({source}) => getStorageBindingCount(source))).toEqual(counts);
  expect(() => requireStorageBudget(requests, parseSources)).toThrow();
});

type ScanSite = {
  exportName: string;
  cases: string[];
  prefix: string;
  scanPrefix?: string;
  consumer: string;
  totalRead: string;
};

const SCAN_SITES: ScanSite[] = [
  {
    exportName: 'GPUNetworkCostWeights',
    cases: ['rows, every column, every output', 'one row, outputs without diagnostics'],
    prefix: 'network-cost-weights',
    consumer: 'network-cost-weights-finalize',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUPolygonRasterization',
    cases: ['every output, single sort', 'every output, two sorts'],
    prefix: 'polygon-rasterization',
    consumer: 'polygon-rasterization-publish',
    totalRead: 'let crossings = total[totalOffset];'
  },
  {
    exportName: 'GPULineDensity',
    cases: ['planar, every output', 'spherical, every output'],
    prefix: 'line-density',
    consumer: 'line-density-publish',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUContiguityWeights',
    cases: ['queen every output large pair capacity', 'rook every output large pair capacity'],
    prefix: 'contiguity-weights-pair',
    consumer: 'contiguity-weights-totals',
    totalRead: 'let pairTotal = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPULatticeWeights',
    cases: ['queen radius 1 mask and cell size', 'rook radius 1 mask and cell size'],
    prefix: 'lattice-weights',
    scanPrefix: 'lattice-weights-scan',
    consumer: 'lattice-weights-offsets',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    cases: ['block with many groups', 'union left rule and distances'],
    prefix: 'spatial-weights-algebra',
    scanPrefix: 'spatial-weights-algebra-scan',
    consumer: 'spatial-weights-algebra-offsets',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUSharedPaths',
    cases: ['every option'],
    prefix: 'shared-paths-vertex',
    consumer: 'shared-paths-finalize',
    totalRead: 'let totalVertices = vertexTotal[vertexTotalOffset];'
  }
];

/** The uint64 scan writes saturated monotone prefixes and the total together. */
function expectSaturatedScanPublication(
  nodes: readonly GPUCommandNode[],
  sources: ReadonlyMap<string, string>,
  scanPrefix: string,
  consumer: string
): number {
  const ids = nodes.map(node => node.id);
  const saturationIndex = ids.indexOf(`${scanPrefix}-saturate`);
  const consumerIndex = ids.indexOf(consumer);
  expect(saturationIndex, ids.join(', ')).toBeGreaterThanOrEqual(0);
  expect(consumerIndex, ids.join(', ')).toBeGreaterThan(saturationIndex);
  const saturatedOutputs = nodes[saturationIndex].resources?.filter(
    resource => resource.usage === 'storage-read-write'
  );
  // Exactly the monotone exclusive offsets and the saturated total are published together.
  expect(saturatedOutputs).toHaveLength(2);
  for (const output of saturatedOutputs!) {
    expect(nodes[consumerIndex].resources).toContainEqual({...output, usage: 'storage-read'});
  }
  const source = sources.get(`${scanPrefix}-saturate`);
  expect(source).toContain(
    'sum = select(low[lowOffset + ROW_COUNT - 1u], 0xffffffffu, high[highOffset + ROW_COUNT - 1u] != 0u);'
  );
  expect(source).toContain(
    'offset = select(low[lowOffset + index - 1u], 0xffffffffu, high[highOffset + index - 1u] != 0u);'
  );
  expect(source).toContain('offsets[offsetsOffset + index] = offset;');
  expect(source).toContain('total[totalOffset + TOTAL_ROW] = sum;');
  return saturationIndex;
}

for (const site of SCAN_SITES) {
  for (const name of site.cases) {
    it(`${site.exportName} saturates its scan total before publishing (${name})`, () => {
      const auditCase = CASES.find(
        candidate => candidate.exportName === site.exportName && candidate.name === name
      );
      expect(auditCase, name).toBeDefined();
      const context = createKernelAuditContext('scan-total-contributors');
      try {
        auditCase!.build(context);
        const nodes = getDeclaredNodes(context) as readonly GPUCommandNode[];
        const ids = nodes.map(node => node.id);
        const sources = requireStorageBudget(nodes);
        if (site.scanPrefix !== undefined) {
          expectSaturatedScanPublication(nodes, sources, site.scanPrefix, site.consumer);
        } else {
          const totalIndex = ids.indexOf(`${site.prefix}-total`);
          const carryIndex = ids.indexOf(`${site.prefix}-carry`);
          expect(totalIndex).toBeGreaterThanOrEqual(0);
          expect(carryIndex).toBe(totalIndex + 1);
          expect(ids.indexOf(site.consumer)).toBeGreaterThan(carryIndex);
          // Total-only consumers still use the older total/carry pair.
          const carryOutput = nodes[carryIndex].resources?.find(
            resource => resource.usage === 'storage-read-write'
          );
          expect(carryOutput).toBeDefined();
          expect(nodes[totalIndex].resources).toContainEqual(carryOutput);
          expect(nodes[ids.indexOf(site.consumer)].resources).toContainEqual({
            ...carryOutput,
            usage: 'storage-read'
          });
          expect(sources.get(`${site.prefix}-carry`)).toContain(
            'atomicStore(&total[totalOffset + TOTAL_ROW], 4294967295u);'
          );
        }
        expect(sources.get(site.consumer)).toContain(site.totalRead);
      } finally {
        context.device.destroy();
      }
    });
  }
}

it('higher-order weights protect every intermediate CSR stage before offsets and overflow', () => {
  const context = createKernelAuditContext('scan-total-higher-order');
  try {
    const auditCase = CASES.find(
      candidate =>
        candidate.exportName === 'GPUSpatialWeightsAlgebra' &&
        candidate.name === 'higherOrder order 2 cumulative'
    );
    expect(auditCase).toBeDefined();
    auditCase!.build(context);
    const nodes = getDeclaredNodes(context) as readonly GPUCommandNode[];
    const ids = nodes.map(node => node.id);
    const sources = getParsedKernelSources(nodes);
    const offsetsIds = ids.filter(id =>
      /^spatial-weights-algebra-(exact|visited)-[0-9]+-offsets$/.test(id)
    );
    expect(offsetsIds).toHaveLength(3);
    for (const offsetsId of offsetsIds) {
      const prefix = offsetsId.slice(0, -'-offsets'.length);
      const saturationIndex = expectSaturatedScanPublication(
        nodes,
        sources,
        `${prefix}-scan`,
        offsetsId
      );
      expect(ids.indexOf(offsetsId)).toBe(saturationIndex + 1);
      expect(sources.get(offsetsId)).toContain('let total = scanTotal[scanTotalOffset];');
    }
  } finally {
    context.device.destroy();
  }
});

it('polygon edge slabs detect carry before clearing counts and disable wrapped indexes', () => {
  const context = createKernelAuditContext('scan-total-slabs');
  try {
    const {nodes} = getPolygonEdgeSlabNodes(context.graph, {
      id: 'slabs',
      operation: 'ScanTotalTest',
      owner: 'polygon',
      positions: context.view('float32x2', 64),
      polygonOffsets: context.view('uint32', 2),
      ringOffsets: context.view('uint32', 2)
    });
    const ids = nodes.map(node => node.id);
    const totalIndex = ids.indexOf('slabs-slab-entry-total');
    const carryIndex = ids.indexOf('slabs-slab-entry-carry');
    expect(totalIndex).toBeGreaterThanOrEqual(0);
    expect(carryIndex).toBe(totalIndex + 1);
    expect(ids[carryIndex + 1]).toBe('slabs-slab-entry-fallback');
    expect(ids.indexOf('slabs-slab-edges-fill')).toBeGreaterThan(carryIndex + 1);
    const sources = getParsedKernelSources(nodes);
    expect(sources.get('slabs-slab-entry-fallback')).toContain(
      'if (scanTotal[scanTotalOffset] == 0xffffffffu)'
    );
    expect(sources.get('slabs-slab-entry-fallback')).toContain(
      'edgeSlabParams[edgeSlabParamsOffset + index * 6u] = 0u;'
    );
    for (const {id, source} of sources.configurations) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
  } finally {
    context.device.destroy();
  }
});

for (const site of [
  ...['planar, banned turns, every output', 'no bans, no arc count'].map(caseName => ({
    exportName: 'GPUNetworkLineGraph',
    caseName,
    scans: ['network-line-graph-scan'],
    consumers: ['network-line-graph-publish']
  })),
  ...['quadbin uncompact depth 15 every option', 'h3 uncompact depth 11 every option'].map(
    caseName => ({
      exportName: 'GPUCellCompaction',
      caseName,
      scans: ['cell-compaction-scan'],
      consumers: ['cell-compaction-expand']
    })
  ),
  {
    exportName: 'GPUNeighborSearch',
    caseName: 'radius planar self join rowStandardize undefined every output',
    scans: ['neighbor-search-scan'],
    consumers: ['neighbor-search-offsets']
  },
  {
    exportName: 'GPUIsobands',
    caseName: 'every output',
    scans: ['isobands-scan', 'isobands-edge-scan'],
    consumers: ['isobands-publish', 'isobands-edge-publish']
  },
  {
    exportName: 'GPUSubdividePolygons',
    caseName: 'rectilinear work queue and clipped bounded pieces',
    scans: ['subdivide-polygons-leaf-scan'],
    consumers: ['subdivide-polygons-rectilinear-status']
  }
]) {
  it(`${site.exportName} uses monotonic saturated offsets and totals before capacity publication${site.exportName === 'GPUCellCompaction' ? ` (${site.caseName})` : ''}`, () => {
    const auditCase = CASES.find(
      candidate =>
        candidate.exportName === site.exportName &&
        (site.caseName === undefined || candidate.name === site.caseName)
    );
    expect(auditCase).toBeDefined();
    const context = createKernelAuditContext('saturated-count-scan-contributors');
    try {
      auditCase!.build(context);
      const nodes = getDeclaredNodes(context) as readonly GPUCommandNode[];
      const ids = nodes.map(node => node.id);
      const sources = requireStorageBudget(nodes, false);
      for (const [index, scan] of site.scans.entries()) {
        const saturationIndex = ids.indexOf(`${scan}-saturate`);
        expect(saturationIndex, ids.join(', ')).toBeGreaterThanOrEqual(0);
        const consumerIndex = ids.indexOf(site.consumers[index]);
        expect(consumerIndex, ids.join(', ')).toBeGreaterThan(saturationIndex);
        const writtenResources = nodes[saturationIndex].resources!.filter(
          resource => resource.usage === 'storage-read-write'
        );
        expect(writtenResources).toHaveLength(2);
        // The publisher reads the saturated total written alongside the monotonic offsets.
        expect(nodes[consumerIndex].resources).toContainEqual({
          ...writtenResources.at(-1),
          usage: 'storage-read'
        });
        expect(sources.get(`${scan}-saturate`)).toContain('high[highOffset + index - 1u] != 0u');
      }
      if (site.exportName === 'GPUCellCompaction') {
        const saturate = nodes[ids.indexOf('cell-compaction-scan-saturate')];
        const expand = nodes[ids.indexOf('cell-compaction-expand')];
        // Expansion binary-searches these same monotone offsets, not the wrapped low words.
        for (const resource of saturate.resources!.filter(
          resource => resource.usage === 'storage-read-write'
        )) {
          expect(expand.resources).toContainEqual({...resource, usage: 'storage-read'});
        }
        const saturationSource = sources.get('cell-compaction-scan-saturate')!;
        expect(saturationSource).toContain(
          'sum = select(low[lowOffset + ROW_COUNT - 1u], 0xffffffffu, high[highOffset + ROW_COUNT - 1u] != 0u);'
        );
        expect(saturationSource).toContain(
          'offset = select(low[lowOffset + index - 1u], 0xffffffffu, high[highOffset + index - 1u] != 0u);'
        );
        expect(saturationSource).toContain('offsets[offsetsOffset + index] = offset;');
        const expansionSource = sources.get('cell-compaction-expand')!;
        expect(expansionSource).toContain('let totalCells = scanTotal[scanTotalOffset];');
        expect(expansionSource).toContain('if (offsets[offsetsOffset + middle] <= index)');
        expect(expansionSource).toContain('totalOut[totalOutOffset] = totalCells;');
        expect(expansionSource).toContain('totalCells > CAPACITY');
      }
      if (site.exportName === 'GPUIsobands') {
        expect(sources.get('isobands-scatter')).toContain('if (start < CAPACITY)');
        expect(sources.get('isobands-edge-scatter')).toContain('if (edgeBase < EDGE_CAPACITY)');
      }
      if (site.exportName === 'GPUSubdividePolygons') {
        expect(sources.get('subdivide-polygons-rectilinear-write')).toContain(
          'getSaturatedScanSum(leafOffsets[leafOffsetsOffset+seed],leaves)'
        );
      }
      for (const {id, source} of sources.configurations) {
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      }
    } finally {
      context.device.destroy();
    }
  });
}
