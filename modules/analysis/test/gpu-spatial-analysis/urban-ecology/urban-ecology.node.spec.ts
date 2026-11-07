// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expect, it} from 'vitest';
import {
  addSegregationCounterfactualRecipe,
  addPopulationSupportAlignmentRecipe,
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addResistanceConnectivityRecipe,
  addSegregationDecompositionRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {
  addStreetProfilesRecipe,
  addIsovistsRecipe,
  addVisibilityGraphRecipe,
  addBuildingTessellationRecipe
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
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
    createTransientView(graph, `urban-ecology-${serial++}`, format, length);
  return {device, graph, view};
}
it('urban and ecology recipes emit parseable graph kernels within portable binding limits', () => {
  const {device, graph, view} = createContext();
  try {
    const patches = view('uint32', 8);
    const patchClasses = view('uint32', 2);
    const positions = view('float32x2', 3);
    const segments = view('float32x4', 8);
    const owners = view('uint32', 8);
    addLandscapeMetricsRecipe(graph, {
      patches,
      patchClasses,
      width: 4,
      height: 2,
      classCount: 2,
      cellSize: [1, 1],
      connectivity: 4
    });
    addPatchProximityRecipe(graph, {
      patches,
      patchClasses,
      width: 4,
      height: 2,
      cellSize: [1, 1],
      edgeDepth: 1,
      searchRadius: 10
    });
    addResistanceConnectivityRecipe(graph, {
      edges: view('uint32x2', 2),
      conductances: view('float32', 2),
      vertexCount: 3,
      source: 0,
      sink: 2
    });
    const populations = view('float32x2', 3);
    addSegregationDecompositionRecipe(graph, {
      populations: [populations, populations, populations, populations],
      supportId: 'common',
      counterfactualMethod: 'supplied',
      randomization: {seed: 5, replicates: 2}
    });
    addStreetProfilesRecipe(graph, {
      positions,
      normals: view('float32x2', 3),
      segments,
      segmentBuildingIds: owners,
      buildingHeights: view('float32', 2),
      maximumDistance: 10,
      sampleSpacing: 1,
      streetSegment: view('float32x4', 1)
    });
    addIsovistsRecipe(graph, {
      positions,
      segments,
      maximumDistance: 10,
      rayCount: 8,
      sampling: 'vertices'
    });
    addVisibilityGraphRecipe(graph, {positions, segments, maximumDistance: 10, edgeCapacity: 8});
    addBuildingTessellationRecipe(graph, {
      segments,
      segmentBuildingIds: owners,
      buildingCount: 2,
      enclosure: view('float32x2', 4),
      enclosureMode: 'simple',
      sampleCapacity: 8,
      sampleSpacing: 1,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 64, rings: 16}
    });
    addSegregationCounterfactualRecipe(graph, {
      baseline: populations,
      final: populations,
      supportId: 'common',
      method: 'composition'
    });
    addPopulationSupportAlignmentRecipe(graph, {
      populations,
      sourceIds: view('uint32', 3),
      targetIds: view('uint32', 3)
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(20);
    for (const kernel of kernels) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});
it('bounded urban/ecology solvers reject unsupported sizes and mismatched shape contracts', () => {
  const {device, graph, view} = createContext();
  try {
    expect(() =>
      addResistanceConnectivityRecipe(graph, {
        vertexCount: 257,
        source: 0,
        sink: 2,
        edges: view('uint32x2', 1),
        conductances: view('float32', 1)
      })
    ).toThrow();
    expect(() =>
      addIsovistsRecipe(graph, {
        positions: view('float32x2', 1),
        segments: view('float32x4', 1),
        maximumDistance: 10,
        rayCount: 2
      })
    ).toThrow();
    expect(() =>
      addSegregationDecompositionRecipe(graph, {
        populations: [
          view('float32x2', 2),
          view('float32x2', 1),
          view('float32x2', 2),
          view('float32x2', 2)
        ],
        supportId: 'common',
        counterfactualMethod: 'supplied',
        randomization: {seed: 5, replicates: 2}
      })
    ).toThrow();
  } finally {
    device.destroy();
  }
});
