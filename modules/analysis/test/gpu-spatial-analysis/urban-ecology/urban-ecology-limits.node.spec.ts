// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  addBuildingAdjacencyRecipe,
  addBuildingTessellationRecipe,
  addIsovistsRecipe,
  addStreetProfilesRecipe,
  addVisibilityGraphRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addPopulationSupportAlignmentRecipe,
  addResistanceConnectivityRecipe,
  addSegregationCounterfactualRecipe,
  addSegregationDecompositionRecipe,
  addSparseResistanceConnectivityRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {
  SPARSE_RESISTANCE_FIXED_COMMAND_NODE_COUNT,
  estimateSparseResistanceCommandNodeCount
} from '../../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-connectivity';
import {URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT} from '../../../src/gpu-spatial-analysis/urban-morphology/urban-ray-utils';
import {GPU_MAXIMUM_COMMAND_NODE_COUNT} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `urban-limits-${serial++}`, format, length);
  return {device, graph, view};
}

function getNodeCount(graph: GPUCommandGraph): number {
  return (graph as unknown as {nodes: unknown[]}).nodes.length;
}

it('urban morphology recipes name the limit they reject', () => {
  const {device, graph, view} = createContext();
  try {
    const segments = view('float32x4', 4);
    const owners = view('uint32', 4);
    const tessellation = {
      segments,
      segmentBuildingIds: owners,
      buildingCount: 2,
      enclosure: view('float32x2', 4),
      sampleCapacity: 8,
      sampleSpacing: 1,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 64, rings: 16}
    };
    expect(() =>
      addBuildingTessellationRecipe(graph, {...tessellation, buildingCount: 65})
    ).toThrow(/^building-tessellation buildingCount must be an integer in \[1, 64\]$/);
    expect(() =>
      addBuildingTessellationRecipe(graph, {...tessellation, enclosure: view('float32x2', 33)})
    ).toThrow(/^building-tessellation enclosure must have between 3 and 32 rows$/);
    expect(() =>
      addBuildingTessellationRecipe(graph, {...tessellation, sampleCapacity: 129})
    ).toThrow(/^building-tessellation sampleCapacity must be an integer in \[1, 128\]$/);
    // Spacing must remain positive and finite after conversion to the shader's float32 type.
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    for (const sampleSpacing of [
      0,
      -1,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      Number.MAX_VALUE,
      Number.MIN_VALUE
    ]) {
      expect(() => addBuildingTessellationRecipe(graph, {...tessellation, sampleSpacing})).toThrow(
        /^building-tessellation sampleSpacing must be a positive finite float32 number$/
      );
      expect(getNodeCount(graph)).toBe(0);
      expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
    }

    const positions = view('float32x2', 2);
    expect(() =>
      addVisibilityGraphRecipe(graph, {
        positions: view('float32x2', 257),
        segments,
        maximumDistance: 1,
        edgeCapacity: 1
      })
    ).toThrow(/^visibility-graph positions must have between 1 and 256 rows$/);
    expect(() =>
      addIsovistsRecipe(graph, {
        positions,
        segments: view('float32x4', 700),
        rayCount: 8,
        sampling: 'vertices',
        maximumDistance: 1
      })
    ).toThrow(/^isovists rayCount plus 6 rays per obstacle segment must be at most 4096/);
    expect(() =>
      addIsovistsRecipe(graph, {positions, segments, rayCount: 2, maximumDistance: 1})
    ).toThrow(/^isovists rayCount must be an integer in \[3, 65536\]$/);
    expect(() =>
      addStreetProfilesRecipe(graph, {
        positions,
        normals: view('float32x2', 1),
        segments,
        segmentBuildingIds: owners,
        buildingHeights: view('float32', 2),
        maximumDistance: 1,
        sampleSpacing: 1
      })
    ).toThrow(/^street-profiles normals must have one row per position$/);
    expect(() =>
      addStreetProfilesRecipe(graph, {
        positions,
        normals: view('float32x2', 2),
        segments: view('float32x4', 1025),
        segmentBuildingIds: view('uint32', 1025),
        buildingHeights: view('float32', 2),
        maximumDistance: 1,
        sampleSpacing: 1,
        streetSegment: view('float32x4', 1)
      })
    ).toThrow(/^street-profiles segments must have at most 1024 rows when streetSegment is given$/);
    expect(() =>
      addBuildingAdjacencyRecipe(graph, {
        sharedPaths: {
          runs: {
            positions: view('float32x2', 2),
            offsets: view('uint32', 2),
            leftSourceIds: view('uint32', 1),
            rightSourceIds: view('uint32', 1),
            forward: view('uint32', 1)
          },
          status: {count: view('uint32', 1)}
        },
        lineBuildingIds: view('uint32', 1),
        orientations: view('float32', 1025)
      })
    ).toThrow(/^building-adjacency orientations must have between 1 and 1024 rows$/);
  } finally {
    device.destroy();
  }
});

it('landscape and segregation recipes name the limit they reject', () => {
  const {device, graph, view} = createContext();
  try {
    expect(() =>
      addLandscapeMetricsRecipe(graph, {
        patches: view('uint32', 2),
        patchClasses: view('uint32', 1),
        width: 2,
        height: 1,
        classCount: 257,
        cellSize: [1, 1],
        connectivity: 4
      })
    ).toThrow(/^landscape-metrics classCount must be an integer in \[1, 256\]$/);
    expect(() =>
      addPatchProximityRecipe(graph, {
        patches: view('uint32', 2),
        patchClasses: view('uint32', 1),
        width: 2,
        height: 2,
        cellSize: [1, 1],
        edgeDepth: 1,
        searchRadius: 1
      })
    ).toThrow(/^patch-proximity patches must have width times height rows$/);
    expect(() =>
      addResistanceConnectivityRecipe(graph, {
        edges: view('uint32x2', 1),
        conductances: view('float32', 1),
        vertexCount: 257,
        source: 0,
        sink: 1
      })
    ).toThrow(/^resistance-connectivity vertexCount must be an integer in \[2, 256\]$/);
    expect(() =>
      addSparseResistanceConnectivityRecipe(graph, {
        edges: view('uint32x2', 1),
        conductances: view('float32', 1),
        vertexCount: 3,
        source: 1,
        sink: 1
      })
    ).toThrow(/^sparse-resistance-connectivity source and sink must be different vertices$/);
    const populations = view('float32x2', 2);
    expect(() =>
      addSegregationDecompositionRecipe(graph, {
        populations: [populations, populations, populations, populations],
        supportId: 'common',
        counterfactualMethod: 'supplied',
        randomization: {seed: 1, replicates: 65537}
      })
    ).toThrow(
      /^segregation-decomposition randomization.replicates must be an integer in \[1, 65536\]$/
    );
    expect(() =>
      addSegregationCounterfactualRecipe(graph, {
        baseline: view('float32x2', 2049),
        final: view('float32x2', 2049),
        supportId: 'common',
        method: 'share'
      })
    ).toThrow(/^segregation-counterfactual baseline rows must be an integer in \[1, 2048\]$/);
    expect(() =>
      addPopulationSupportAlignmentRecipe(graph, {
        populations,
        sourceIds: view('uint32', 3),
        targetIds: view('uint32', 2)
      })
    ).toThrow(/^population-support-alignment sourceIds must have 2 rows$/);
  } finally {
    device.destroy();
  }
});

it('the sparse resistance estimate is never below the command nodes the recipe adds', () => {
  // Edge counts 2^19 apart: the published sort and scan add nodes as the counts grow.
  for (const [edgeCount, vertexCount, maximumIterations, connectivityIterations] of [
    [4, 1048576, 1, 1],
    [4, 4, 7, 3],
    [4, 1048576, 64, 16],
    [4, 1048576, 1024, 128],
    [4096, 1048576, 7, 3],
    [2097152, 1048576, 1, 1],
    [2097152, 4, 64, 16],
    [2097152, 1048576, 1024, 128]
  ] as const) {
    const {device, graph, view} = createContext();
    try {
      const before = getNodeCount(graph);
      const result = addSparseResistanceConnectivityRecipe(graph, {
        edges: view('uint32x2', edgeCount),
        conductances: view('float32', edgeCount),
        vertexCount,
        source: 0,
        sink: 1,
        maximumIterations,
        connectivityIterations
      });
      const actual = getNodeCount(graph) - before;
      expect(result.estimatedCommandNodeCount).toBe(
        estimateSparseResistanceCommandNodeCount(maximumIterations, connectivityIterations)
      );
      expect(result.estimatedCommandNodeCount, `${edgeCount} edges`).toBeGreaterThanOrEqual(actual);
      // The allowance stays tight enough to be informative.
      expect(result.estimatedCommandNodeCount - actual, `${edgeCount} edges`).toBeLessThan(60);
    } finally {
      device.destroy();
    }
  }
});

it('the largest sparse resistance configuration fits the default command node budget', () => {
  expect(estimateSparseResistanceCommandNodeCount(4096, 1024)).toBeLessThanOrEqual(
    GPU_MAXIMUM_COMMAND_NODE_COUNT
  );
  expect(SPARSE_RESISTANCE_FIXED_COMMAND_NODE_COUNT).toBeGreaterThan(0);
});

it('ray recipes switch to the segment BVH above the brute-force cap and emit auditable kernels', () => {
  const {device, graph, view} = createContext();
  try {
    const small = URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT;
    const positions = view('float32x2', 2);
    const common = {positions, maximumDistance: 10, rayCount: 8};
    expect(
      addIsovistsRecipe(graph, {...common, id: 'at-cap', segments: view('float32x4', small)})
        .provenance.obstacleIndex
    ).toBe('brute-force');
    const large = addIsovistsRecipe(graph, {
      ...common,
      id: 'above-cap',
      segments: view('float32x4', small + 1)
    });
    expect(large.provenance.obstacleIndex).toBe('bvh');
    expect(
      addIsovistsRecipe(graph, {
        ...common,
        id: 'forced-brute',
        segments: view('float32x4', small + 1),
        obstacleIndex: 'brute-force'
      }).provenance.obstacleIndex
    ).toBe('brute-force');
    const forced = addIsovistsRecipe(graph, {
      ...common,
      id: 'forced-bvh',
      segments: view('float32x4', 8),
      obstacleIndex: 'bvh'
    });
    expect(forced.provenance.obstacleIndex).toBe('bvh');
    const profiles = addStreetProfilesRecipe(graph, {
      id: 'profiles-bvh',
      positions,
      normals: view('float32x2', 2),
      segments: view('float32x4', 8),
      segmentBuildingIds: view('uint32', 8),
      buildingHeights: view('float32', 2),
      maximumDistance: 10,
      sampleSpacing: 1,
      streetSegment: view('float32x4', 1),
      obstacleIndex: 'bvh'
    });
    expect(profiles.provenance.obstacleIndex).toBe('bvh');
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(20);
    for (const kernel of kernels) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      // The sorted BVH build reuses a bounds reduction whose source the WGSL reflection parser rejects.
      if (!kernel.id.includes('sort-scene-bounds-blocks')) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      }
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
