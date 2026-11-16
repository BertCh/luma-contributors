// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type VertexFormat} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validateUrbanRecipeBindings} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-validation';
import {expectSpatialAnalysisWGSL} from '../geographic-distribution/spatial-analysis-wgsl-test-utils';
import {publishBuildingTessellationOutput} from './building-tessellation-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {WgslReflect} from 'wgsl_reflect';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {
  BUILDING_TESSELLATION_LOCAL_COMMAND_NODE_ALLOWANCE,
  BUILDING_TESSELLATION_LOCAL_MAXIMUM_BUILDING_COUNT
} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation-local';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createContext(buildingCount = 100000) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `input-${serial++}`, format, length);
  const props = {
    segments: view('float32x4', buildingCount * 4),
    segmentBuildingIds: view('uint32', buildingCount * 4),
    enclosure: view('float32x2', 4),
    buildingCount,
    sampleSpacing: 1,
    sampleCapacity: buildingCount * 4,
    candidateRadius: 2.75,
    vertexTolerance: 1 / 1024
  };
  return {device, graph, view, props};
}
function getNodes(graph: GPUCommandGraph) {
  return (graph as unknown as {nodes: {id: string}[]}).nodes;
}

it('100k buildings use one clipping dispatch and bindings bounded by 128 MiB', () => {
  const {device, graph, props} = createContext();
  try {
    const result = addBuildingTessellationRecipe(graph, props);
    expect(result.provenance.mode).toBe('local');
    expect(result.outputs.positions.length * 8).toBe(76800000);
    expect(result.outputs.cells.length).toBe(800000);
    const nodes = getNodes(graph);
    expect(nodes.filter(node => node.id === 'building-tessellation-local-cells')).toHaveLength(1);
    expect(nodes.some(node => /building-\d+/.test(node.id))).toBe(false);
    expect(nodes.length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    expect(result.estimatedCommandNodeCount).toBe(
      BUILDING_TESSELLATION_LOCAL_COMMAND_NODE_ALLOWANCE
    );
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const clip = kernels.find(kernel => kernel.id === 'building-tessellation-local-cells')!;
    expect(clip).toBeDefined();
    expect(clip.source).toContain('let first=buildingIndex');
    expect(getStorageBindingCount(clip.source)).toBe(8);
    for (const kernel of kernels) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('the command allowance bounds the largest sample count and million-building topology', () => {
  const {device, graph, props} = createContext(BUILDING_TESSELLATION_LOCAL_MAXIMUM_BUILDING_COUNT);
  try {
    // This topology requests higher binding limits for packed sample data; no GPU buffers are allocated.
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: 1073741824,
      maxBufferSize: 1073741824
    });
    const result = addBuildingTessellationRecipe(graph, {
      ...props,
      sampleCapacity: 16777216,
      maximumSamplesPerBuilding: 1,
      cellVertexCapacity: 3
    });
    expect(getNodes(graph).length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});

it('reference mode keeps its named 64-building and 128-sample limits', () => {
  const {device, graph, props} = createContext(2);
  try {
    const reference = {
      ...props,
      mode: 'reference' as const,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 128, rings: 16}
    };
    expect(() => addBuildingTessellationRecipe(graph, {...reference, buildingCount: 65})).toThrow(
      /^building-tessellation buildingCount must be an integer in \[1, 64\]$/
    );
    expect(() => addBuildingTessellationRecipe(graph, {...reference, sampleCapacity: 129})).toThrow(
      /^building-tessellation sampleCapacity must be an integer in \[1, 128\]$/
    );
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('local limits and output binding sizes fail before any command nodes are added', () => {
  const {device, graph, props} = createContext();
  try {
    expect(() => addBuildingTessellationRecipe(graph, {...props, candidateCapacity: 257})).toThrow(
      /^building-tessellation candidateCapacity must be an integer in \[1, 256\]$/
    );
    expect(() =>
      addBuildingTessellationRecipe(graph, {...props, maximumSamplesPerBuilding: 33})
    ).toThrow(/^building-tessellation maximumSamplesPerBuilding must be an integer in \[1, 32\]$/);
    expect(() => addBuildingTessellationRecipe(graph, {...props, maximumGridVisits: 8193})).toThrow(
      /^building-tessellation maximumGridVisits must be an integer in \[1, 8192\]$/
    );
    expect(() => addBuildingTessellationRecipe(graph, {...props, cellVertexCapacity: 128})).toThrow(
      /^building-tessellation positions needs 819200000 bytes in one binding/
    );
    expect(() =>
      addBuildingTessellationRecipe(graph, {
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
    ).toThrow(/does not support longitude-latitude coordinates/);
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('validates aligned binding sizes and aliasing before graph mutation', () => {
  const {device, graph, props} = createContext(524288);
  try {
    const storage = graph.createTransientBuffer({
      id: 'misaligned-result',
      byteLength: 134217732,
      usage: Buffer.STORAGE
    });
    const positions = graph.createDataView(storage, {
      format: 'float32x2',
      byteOffset: 4,
      length: 16777216
    });
    const resourceCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() =>
      addBuildingTessellationRecipe(graph, {
        ...props,
        maximumSamplesPerBuilding: 2,
        cellVertexCapacity: 16,
        output: {positions}
      })
    ).toThrow(/needs 134217732 bytes/);
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(resourceCount);
    const shared = createTransientView(graph, 'shared-status', 'uint32', 64);
    const first = graph.createDataView(shared.buffer, {format: 'uint32', length: 1});
    const second = graph.createDataView(shared.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    expect(() =>
      addBuildingTessellationRecipe(graph, {
        ...props,
        buildingCount: 1,
        output: {status: {count: first, requiredCount: second}}
      })
    ).toThrow(/logical byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('distinct imported handles cannot hide physical-buffer output aliases', () => {
  const {device, graph, props} = createContext(2);
  const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'first-import', byteLength: 512, usage: Buffer.STORAGE},
      buffer
    );
    const second = graph.importBuffer(
      {id: 'second-import', byteLength: 512, usage: Buffer.STORAGE},
      buffer
    );
    expect(() =>
      addBuildingTessellationRecipe(graph, {
        ...props,
        output: {
          status: {
            count: graph.createDataView(first, {format: 'uint32', length: 1}),
            requiredCount: graph.createDataView(second, {
              format: 'uint32',
              byteOffset: 0,
              length: 1
            })
          }
        }
      })
    ).toThrow(/logical byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('preflight allows consecutive writable chunks in one alignment window but rejects overlapping writes', () => {
  const {device, graph} = createContext(1);
  try {
    const shared = createTransientView(graph, 'chunked-output', 'uint32', 4);
    const first = graph.createDataView(shared.buffer, {format: 'uint32', length: 1});
    const second = graph.createDataView(shared.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    const vector = (data: (typeof first)[]) =>
      new GraphVectorView({
        id: 'chunks',
        name: 'chunks',
        format: 'uint32',
        length: data.length,
        valueLength: data.length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data
      });
    const resourceCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() =>
      validateUrbanRecipeBindings(graph, 'urban-preflight', {}, {values: vector([first, second])})
    ).not.toThrow();
    expect(() =>
      validateUrbanRecipeBindings(graph, 'urban-preflight', {}, {values: vector([first, first])})
    ).toThrow(/logical byte ranges must not overlap/);
    expect(() =>
      validateUrbanRecipeBindings(graph, 'urban-preflight', {}, {first, second})
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(resourceCount);
  } finally {
    device.destroy();
  }
});

it('the shared headless publisher does not reuse the reference recipe resource IDs', () => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, 'readback-resource-ids');
  try {
    const graph = fixture.graph;
    const reference = addBuildingTessellationRecipe(graph, {
      id: 'reference',
      mode: 'reference',
      segments: createTransientView(graph, 'segments', 'float32x4', 8),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 8),
      buildingCount: 2,
      enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
      sampleSpacing: 1,
      sampleCapacity: 8,
      vertexTolerance: 1 / 1024,
      searchStructure: 'brute-force',
      capacity: {intersections: 256, nodedSegments: 256, vertices: 128, rings: 32}
    });
    expect(reference.status.complete.buffer.id).toBe('reference-complete');
    const complete = publishBuildingTessellationOutput(
      fixture,
      'reference-complete',
      reference.status.complete
    );
    expect(complete.view.buffer.id).toBe('readback-reference-complete');
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it.each([
  1e20, 1e21, 3.4028234663852886e38
])('local shader numeric audit covers extreme finite scalar props: %s', value => {
  const {device, graph, props} = createContext(2);
  try {
    addBuildingTessellationRecipe(graph, {
      ...props,
      sampleSpacing: value,
      candidateRadius: value,
      vertexTolerance: value
    });
    for (const kernel of getGPUCommandNodeKernelSources(getNodes(graph)))
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it('local tessellation checks maxBufferSize independently of a raised storage-binding limit', () => {
  const {device, graph, props} = createContext(1000000);
  try {
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: 1073741824,
      maxBufferSize: 268435456
    });
    const bufferCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    expect(() => addBuildingTessellationRecipe(graph, props)).toThrow(
      /building-tessellation positions allocation byteLength \(maxBufferSize\).*268435456/
    );
    expect(getNodes(graph)).toHaveLength(0);
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(bufferCount);
  } finally {
    device.destroy();
  }
});
