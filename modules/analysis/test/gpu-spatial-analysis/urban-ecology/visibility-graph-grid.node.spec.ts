// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getGPUCommandNodePipelineKeys} from '../../utils/gpu-contributor-test-utils';
import {
  expectUrbanWGSLIdentifiers,
  expectUrbanWGSLIntegerLiterals
} from './visibility-graph-wgsl-test-utils';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {addVisibilityGraphRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {addUrbanPointGrid} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-point-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function getNodes(graph: GPUCommandGraph): {id: string}[] {
  return (graph as unknown as {nodes: {id: string}[]}).nodes;
}

it.each([
  'grid',
  'brute-force'
] as const)('visibility preserves the explicit %s candidate route and the eight-binding budget', searchStructure => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addVisibilityGraphRecipe(graph, {
      positions: createTransientView(graph, 'points', 'float32x2', 32),
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      maximumDistance: 2,
      edgeCapacity: 32,
      searchStructure,
      obstacleIndex: 'bvh'
    });
    expect(result.provenance.searchStructure).toBe(searchStructure);
    const kernels = getGPUCommandNodeKernelSources(getNodes(graph));
    const visibility = kernels.find(kernel => kernel.id === 'visibility-graph-visibility')!;
    expect(visibility).toBeDefined();
    expect(visibility.source.includes('getUrbanPointGridRow')).toBe(searchStructure === 'grid');
    for (const kernel of kernels) {
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      if (!kernel.id.includes('sort-scene-bounds-blocks')) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      }
    }
  } finally {
    device.destroy();
  }
});

it('urban point grids accept ten thousand rows without quadratic scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const grid = addUrbanPointGrid(graph, {
      id: 'city-grid',
      positions: createTransientView(graph, 'city-points', 'float32x2', 10000)
    });
    expect(grid.data.length).toBe(20005);
    expect(grid.dimension).toBe(100);
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph))) {
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'grid',
  'brute-force'
] as const)('tessellation keeps the %s clip route with bounded shader bindings', searchStructure => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addBuildingTessellationRecipe(graph, {
      segments: createTransientView(graph, 'walls', 'float32x4', 8),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 8),
      buildingCount: 2,
      enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
      sampleSpacing: 1,
      sampleCapacity: 16,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16},
      searchStructure
    });
    expect(result.provenance.searchStructure).toBe(searchStructure);
    const kernels = getGPUCommandNodeKernelSources(getNodes(graph));
    const clip = kernels.find(kernel => kernel.id.endsWith('clip-cells'))!;
    expect(clip).toBeDefined();
    expect(clip.source.includes('gridRadius')).toBe(searchStructure === 'grid');
    for (const kernel of kernels) {
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('recipes reject oversized bindings before adding nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const segments = createTransientView(graph, 'walls', 'float32x4', 4);
    expect(() =>
      addVisibilityGraphRecipe(graph, {
        positions: createTransientView(graph, 'points', 'float32x2', 2),
        segments,
        maximumDistance: 1,
        edgeCapacity: 16777217
      })
    ).toThrow(/^visibility-graph edges needs 134217736 bytes in one binding/);
    expect(getNodes(graph)).toHaveLength(0);
    expect(() =>
      addBuildingTessellationRecipe(graph, {
        segments,
        segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 4),
        buildingCount: 1,
        enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
        sampleSpacing: 1,
        sampleCapacity: 8,
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 1, nodedSegments: 8388609, vertices: 8, rings: 1}
      })
    ).toThrow(/^building-tessellation overlay boundary needs 134217744 bytes in one binding/);
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('reference building pack/evidence pipelines do not specialize on building IDs', () => {
  for (const buildingCount of [2, 5]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      addBuildingTessellationRecipe(graph, {
        segments: createTransientView(graph, 'walls', 'float32x4', 8),
        segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 8),
        buildingCount,
        enclosure: createTransientView(graph, 'study', 'float32x2', 4),
        sampleSpacing: 1,
        sampleCapacity: 8,
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16}
      });
      const nodes = getNodes(graph).filter(node => /building-\d+-(pack|evidence)$/.test(node.id));
      expect(nodes).toHaveLength(2 * buildingCount);
      expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(2);
      expect(new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source)).size).toBe(
        2
      );
    } finally {
      device.destroy();
    }
  }
});

it.each([
  undefined,
  'reference'
] as const)('reference call mode %s defaults to exhaustive sample order', mode => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const result = addBuildingTessellationRecipe(graph, {
      mode,
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 4),
      buildingCount: 1,
      enclosure: createTransientView(graph, 'study', 'float32x2', 4),
      sampleSpacing: 2,
      sampleCapacity: 4,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16}
    });
    expect(result.provenance.searchStructure).toBe('brute-force');
    expect(getNodes(graph).some(node => node.id.includes('sample-grid'))).toBe(false);
    const clip = getGPUCommandNodeKernelSources(
      getNodes(graph).filter(node => node.id.endsWith('clip-cells'))
    )[0];
    expect(clip.source).not.toContain('getUrbanPointGridRow');
    expectUrbanWGSLIdentifiers(clip.source, clip.id);
  } finally {
    device.destroy();
  }
});

// These values reproduce the Metal parse failure: JS prints 1e20 as an integer token.
it.each([
  'grid',
  'brute-force'
] as const)('large float props produce legal float literals in both reference recipes: %s', searchStructure => {
  const device = createNullWebGPUDevice();
  try {
    const tessellation = new GPUCommandGraph(device);
    addBuildingTessellationRecipe(tessellation, {
      segments: createTransientView(tessellation, 'walls', 'float32x4', 8),
      segmentBuildingIds: createTransientView(tessellation, 'owners', 'uint32', 8),
      buildingCount: 2,
      enclosure: createTransientView(tessellation, 'enclosure', 'float32x2', 4),
      sampleSpacing: 1e20,
      sampleCapacity: 8,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16},
      searchStructure
    });
    const visibility = new GPUCommandGraph(device);
    addVisibilityGraphRecipe(visibility, {
      positions: createTransientView(visibility, 'points', 'float32x2', 4),
      segments: createTransientView(visibility, 'walls', 'float32x4', 4),
      maximumDistance: 1e20,
      tolerance: 1 / 1024,
      edgeCapacity: 6,
      searchStructure,
      obstacleIndex: 'bvh'
    });
    const kernels = getGPUCommandNodeKernelSources(getNodes(tessellation));
    const sampling = kernels.find(kernel => kernel.id.endsWith('-sample-counts'))!;
    // The correctly rounded f32 value must retain a float spelling even without an exponent.
    expect(sampling.source).toContain('ceil(length/100000002004087730000.0)');
    for (const kernel of [...kernels, ...getGPUCommandNodeKernelSources(getNodes(visibility))]) {
      expectUrbanWGSLIdentifiers(kernel.source, kernel.id);
    }
  } finally {
    device.destroy();
  }
});

it('the literal audit catches the Metal failure even when reflection accepts the source', () => {
  const invalid = 'fn countSamples(length:f32)->f32{return ceil(length/100000000000000000000);}';
  expect(() => new WgslReflect(invalid)).not.toThrow();
  expect(() => expectUrbanWGSLIntegerLiterals(invalid, 'regression')).toThrow();
  expect(() => expectUrbanWGSLIntegerLiterals('let row=4294967296u;', 'u32 overflow')).toThrow();
  expect(() => expectUrbanWGSLIntegerLiterals('let row=2147483648i;', 'i32 overflow')).toThrow();
  expectUrbanWGSLIntegerLiterals(
    'let row123456789012345678901 = 0xffffffffu; let spacing = 100000002004087730000.0; let exponent = 1e20; /* 999999999999999999999 */',
    'valid integers, floats, identifiers and comments'
  );
});

it.each([
  Number.MAX_VALUE,
  Number.MIN_VALUE
])('unrepresentable positive float props fail before graph mutation: %s', value => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const tessellation = {
      segments: createTransientView(graph, 'walls', 'float32x4', 4),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 4),
      buildingCount: 1,
      enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
      sampleSpacing: 1,
      sampleCapacity: 4,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16}
    };
    const visibility = {
      positions: createTransientView(graph, 'points', 'float32x2', 4),
      segments: tessellation.segments,
      maximumDistance: 1,
      tolerance: 1 / 1024,
      edgeCapacity: 6
    };
    const buffers = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    for (const field of ['sampleSpacing', 'vertexTolerance'] as const) {
      expect(() => addBuildingTessellationRecipe(graph, {...tessellation, [field]: value})).toThrow(
        new RegExp(`building-tessellation.*${field}.*float32`)
      );
    }
    for (const field of ['maximumDistance', 'tolerance'] as const) {
      expect(() => addVisibilityGraphRecipe(graph, {...visibility, [field]: value})).toThrow(
        new RegExp(`visibility-graph.*${field}.*float32`)
      );
    }
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(buffers);
  } finally {
    device.destroy();
  }
});
