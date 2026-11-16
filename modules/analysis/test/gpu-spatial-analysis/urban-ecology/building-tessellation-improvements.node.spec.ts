// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {BUILDING_TESSELLATION_LOCAL_COMMAND_NODE_ALLOWANCE} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-local';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {expectSpatialAnalysisWGSL} from '../geographic-distribution/spatial-analysis-wgsl-test-utils';
import {
  CLUSTERED_TESSELLATION_COLUMNS,
  createClusteredTessellationInput,
  getClusteredTessellationOracle
} from './building-tessellation-cluster-test-utils';

it.each([
  [10000, 9801],
  [100000, 98901]
])('%i clustered buildings have %i certifiable rectangles despite the outlier', (buildingCount, certifiedCount) => {
  const oracle = getClusteredTessellationOracle(buildingCount);
  const {segments} = createClusteredTessellationInput(buildingCount);
  expect(oracle.status).toEqual({
    count: certifiedCount,
    requiredCount: buildingCount,
    overflow: 0,
    candidateOverflow: 1,
    invalidCount: 0
  });
  let predictedCount = 0;
  for (let building = 0; building < buildingCount; building++)
    predictedCount += Number(oracle.isCertified(building));
  expect(predictedCount).toBe(certifiedCount);
  expect(oracle.isCertified(buildingCount - 1)).toBe(false);
  expect(Array.from(segments.slice(-16, -14))).toEqual([1e6, 1e6]);
  const lastCertified = buildingCount - CLUSTERED_TESSELLATION_COLUMNS - 2;
  for (const building of [0, 101, lastCertified]) {
    expect(oracle.isCertified(building)).toBe(true);
    const {lower, upper} = oracle.bounds(building);
    const center = [segments[16 * building] + 0.0625, segments[16 * building + 1] + 0.0625];
    const slack = Math.max(4 / 1024, 1e-5 * (Math.max(...center) + 2.75));
    let area = 0;
    for (let corner = 0; corner < 4; corner++) {
      const site = [segments[16 * building + 4 * corner], segments[16 * building + 4 * corner + 1]];
      const cellLower = site.map((coordinate, axis) =>
        coordinate < center[axis] ? lower[axis] : center[axis]
      );
      const cellUpper = site.map((coordinate, axis) =>
        coordinate < center[axis] ? center[axis] : upper[axis]
      );
      const greatestDistance = Math.hypot(
        Math.max(site[0] - cellLower[0], cellUpper[0] - site[0]),
        Math.max(site[1] - cellLower[1], cellUpper[1] - site[1])
      );
      // This is the documented triangle-inequality certificate, applied to closed-form cells.
      expect(
        Math.hypot(site[0] - center[0], site[1] - center[1]) + 2 * greatestDistance + slack
      ).toBeLessThan(2.75);
      area += (cellUpper[0] - cellLower[0]) * (cellUpper[1] - cellLower[1]);
    }
    expect(area).toBe(building === 0 ? 0.8125 ** 2 : 1);
  }
});

it('cluster rectangle boundaries agree with exhaustive nearest-sample distances', () => {
  const buildingCount = 300;
  const {segments} = createClusteredTessellationInput(buildingCount);
  const oracle = getClusteredTessellationOracle(buildingCount);
  function nearestSquared(point: number[], owner?: number) {
    let nearest = Infinity;
    for (let building = 0; building < buildingCount; building++) {
      if (owner !== undefined && building !== owner) continue;
      for (let corner = 0; corner < 4; corner++) {
        const offset = 16 * building + 4 * corner;
        nearest = Math.min(
          nearest,
          (point[0] - segments[offset]) ** 2 + (point[1] - segments[offset + 1]) ** 2
        );
      }
    }
    return nearest;
  }
  for (const building of [0, 101, 198]) {
    const {lower, upper} = oracle.bounds(building);
    for (const horizontal of [lower[0], upper[0]])
      for (const vertical of [lower[1], upper[1]]) {
        const point = [horizontal, vertical];
        expect(nearestSquared(point, building)).toBe(nearestSquared(point));
      }
    // Crossing either outer edge puts the probe strictly nearer another building's samples.
    for (const point of [
      [upper[0] + 0.001, (lower[1] + upper[1]) / 2],
      [(lower[0] + upper[0]) / 2, upper[1] + 0.001]
    ])
      expect(nearestSquared(point, building)).toBeGreaterThan(nearestSquared(point));
  }
});

function createContext(buildingCount = 2) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    segments: createTransientView(graph, 'segments', 'float32x4', buildingCount * 4),
    segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', buildingCount * 4),
    enclosure: createTransientView(graph, 'enclosure', 'float32x2', 8),
    enclosureRingOffsets: createTransientView(graph, 'rings', 'uint32', 3),
    enclosureMode: 'simple' as const,
    buildingCount,
    sampleSpacing: 1,
    sampleCapacity: buildingCount * 4,
    candidateRadius: 32,
    vertexTolerance: 1 / 1024
  };
  return {device, graph, props};
}
function getGraphShape(graph: GPUCommandGraph) {
  const state = graph as unknown as {nodes: {id: string}[]; buffers: Map<string, unknown>};
  return {nodes: state.nodes, resourceCount: state.buffers.size};
}

it('simple enclosures with holes use one bounded clipping dispatch and audited WGSL', () => {
  const {device, graph, props} = createContext(1000);
  try {
    const result = addBuildingTessellationRecipe(graph, props);
    expect(result.provenance.enclosurePieceCount).toBe(8);
    expect(result.provenance.piecesPerBuilding).toBe(64);
    expect(result.outputs.cells.length).toBe(64000);
    expect(result.outputs.positions.length).toBe(768000);
    expect(getGraphShape(graph).nodes.length).toBeLessThanOrEqual(
      BUILDING_TESSELLATION_LOCAL_COMMAND_NODE_ALLOWANCE
    );
    const kernels = getGPUCommandNodeKernelSources(getGraphShape(graph).nodes);
    expect(
      kernels.filter(kernel => kernel.id === 'building-tessellation-local-cells')
    ).toHaveLength(1);
    expect(kernels.some(kernel => kernel.operation === 'GPUPolygonTriangulation')).toBe(true);
    for (const kernel of kernels) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('clipping and index shader counts stay constant as the building query count grows', () => {
  const counts = [1, 1000].map(buildingCount => {
    const {device, graph, props} = createContext(buildingCount);
    try {
      addBuildingTessellationRecipe(graph, props);
      const kernels = getGPUCommandNodeKernelSources(getGraphShape(graph).nodes);
      const clipping = kernels.filter(kernel => kernel.id === 'building-tessellation-local-cells');
      const index = kernels.filter(kernel => kernel.operation === 'BuildingTessellationIndex');
      return [
        new Set(clipping.map(kernel => kernel.source)).size,
        new Set(index.map(kernel => kernel.source)).size
      ];
    } finally {
      device.destroy();
    }
  });
  expect(counts).toEqual([
    [1, 7],
    [1, 7]
  ]);
});

it('checks all supplied output shapes and enclosure topology before allocating missing outputs', () => {
  const {device, graph, props} = createContext();
  try {
    const cells = createTransientView(graph, 'short-cells', 'uint32x4', 1);
    const before = getGraphShape(graph).resourceCount;
    expect(() => addBuildingTessellationRecipe(graph, {...props, output: {cells}})).toThrow(
      /output.cells/
    );
    expect(getGraphShape(graph).nodes).toHaveLength(0);
    expect(getGraphShape(graph).resourceCount).toBe(before);
    expect(() => addBuildingTessellationRecipe(graph, {...props, enclosureMode: 'convex'})).toThrow(
      /enclosureRingOffsets requires enclosureMode 'simple'/
    );
    expect(getGraphShape(graph).resourceCount).toBe(before);
    expect(getGraphShape(graph).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('checks the packed coordinate index binding before allocating any resources', () => {
  const {device, graph, props} = createContext();
  try {
    const before = getGraphShape(graph).resourceCount;
    expect(() => addBuildingTessellationRecipe(graph, {...props, sampleCapacity: 8388609})).toThrow(
      /sorted coordinate index needs 171966488 bytes/
    );
    expect(getGraphShape(graph).nodes).toHaveLength(0);
    expect(getGraphShape(graph).resourceCount).toBe(before);
  } finally {
    device.destroy();
  }
});

it('reference boundary sampling saturates every prefix before writing samples', () => {
  const {device, graph, props} = createContext();
  try {
    addBuildingTessellationRecipe(graph, {
      ...props,
      mode: 'reference',
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16}
    });
    const kernels = getGPUCommandNodeKernelSources(getGraphShape(graph).nodes);
    const saturation = kernels.find(
      kernel => kernel.id === 'building-tessellation-sample-total-saturate'
    );
    expect(saturation).toBeDefined();
    expect(saturation!.source).toContain('offsets[offsetsOffset + index] = offset;');
    expect(saturation!.source).toContain('0xffffffffu');
    expect(kernels.some(kernel => kernel.id === 'building-tessellation-sample-scan')).toBe(false);
  } finally {
    device.destroy();
  }
});

it('local scalar status fields in separate dispatches may occupy one packed alignment window', () => {
  const {device, graph, props} = createContext();
  try {
    const buffer = createTransientView(graph, 'packed-status', 'uint32', 8).buffer;
    const names = [
      'count',
      'requiredCount',
      'overflow',
      'candidateOverflow',
      'invalidCount'
    ] as const;
    const status = Object.fromEntries(
      names.map((name, index) => [
        name,
        graph.createDataView(buffer, {format: 'uint32', byteOffset: index * 4, length: 1})
      ])
    );
    expect(() => addBuildingTessellationRecipe(graph, {...props, output: {status}})).not.toThrow();
  } finally {
    device.destroy();
  }
});
