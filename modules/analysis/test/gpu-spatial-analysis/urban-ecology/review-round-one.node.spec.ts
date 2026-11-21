// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addSegregationDecompositionRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-decomposition';
import {addBuildingAdjacencyRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-adjacency';
import {addBuildingTessellationReferenceRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {addStreetProfilesRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/street-profiles';
import {addPatchProximityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/patch-proximity';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/landscape-metrics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function inventory(graph: GPUCommandGraph) {
  const state = graph as unknown as {nodes: {id: string}[]; buffers: Map<string, unknown>};
  return {nodes: state.nodes.map(node => node.id), buffers: [...state.buffers.keys()]};
}

it.each([
  'swaps',
  'inference'
] as const)('B1 preflights %s device bindings before mutation and same-ID retry', failure => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const population = createTransientView(graph, 'population', 'float32x2', 1024);
    const swaps = createTransientView(graph, 'large-swaps', 'uint32', 1024 * 40000);
    const buffer = graph.importBuffer({id: 'bad-usage', byteLength: 16, usage: Buffer.COPY_DST});
    const inference = graph.createDataView(buffer, {format: 'float32x4', length: 1});
    const props = {
      id: 'large-swaps',
      populations: [population, population, population, population] as const,
      supportId: 'support',
      counterfactualMethod: 'supplied'
    };
    const before = inventory(graph);
    let rejection: unknown;
    try {
      addSegregationDecompositionRecipe(graph, {
        ...props,
        randomization:
          failure === 'swaps' ? {replicates: 40000, seed: 1, swaps} : {replicates: 4, seed: 1},
        output: failure === 'inference' ? {inference} : undefined
      });
    } catch (error) {
      rejection = error;
    }
    expect(String(rejection)).toMatch(failure === 'swaps' ? /binding|bytes|134217728/ : /storage/i);
    expect(inventory(graph)).toEqual(before);
    addSegregationDecompositionRecipe(graph, {...props, randomization: {replicates: 4, seed: 1}});
    expect(inventory(graph).nodes.length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('B3 preflights every street output before output reservation and allows same-ID recovery', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      id: 'street-retry',
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      normals: createTransientView(graph, 'normals', 'float32x2', 2),
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 1),
      buildingHeights: createTransientView(graph, 'heights', 'float32', 1),
      maximumDistance: 5,
      sampleSpacing: 1
    };
    const validity = createTransientView(graph, 'short', 'uint32', 1);
    const before = inventory(graph);
    expect(() => addStreetProfilesRecipe(graph, {...props, output: {validity}})).toThrow(/rows/);
    expect(inventory(graph)).toEqual(before);
    addStreetProfilesRecipe(graph, props);
    expect(inventory(graph).nodes.length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

function adjacencyProps(graph: GPUCommandGraph) {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  return {
    sharedPaths: {
      runs: {
        positions: view('vertices', 'float32x2', 2),
        offsets: view('offsets', 'uint32', 2),
        leftSourceIds: view('left', 'uint32', 1),
        rightSourceIds: view('right', 'uint32', 1),
        forward: view('forward', 'uint32', 1)
      },
      status: {
        count: view('count', 'uint32', 1),
        requiredCount: view('required', 'uint32', 1),
        candidateOverflow: view('candidate', 'uint32', 1),
        vertexCount: view('vertex-count', 'uint32', 1),
        requiredVertexCount: view('vertex-required', 'uint32', 1)
      }
    },
    lineBuildingIds: view('owners', 'uint32', 2),
    orientations: view('angles', 'float32', 2)
  };
}

it('B2 adjacency consumes candidate loss and run/vertex requirements without optional overflow', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    addBuildingAdjacencyRecipe(graph, adjacencyProps(graph));
    const evidence = getGPUCommandNodeKernelSources(
      (graph as unknown as {nodes: unknown[]}).nodes
    ).find(kernel => kernel.id.endsWith('-evidence'))!;
    for (const name of ['candidateOverflow', 'requiredCount', 'vertexCount', 'requiredVertexCount'])
      expect(evidence.source).toContain(`${name}[${name}Offset]`);
  } finally {
    device.destroy();
  }
});

it('earlier-fix note: adjacency ignores structural caller metadata', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = adjacencyProps(graph);
    const sharedPaths = {
      ...props.sharedPaths,
      status: {...props.sharedPaths.status, label: 'provided'},
      runs: {...props.sharedPaths.runs, label: 'provided'}
    };
    expect(() => addBuildingAdjacencyRecipe(graph, {...props, sharedPaths})).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('earlier-fix note: landscape rejects float32-underflow cell widths before mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      patches: createTransientView(graph, 'patches', 'uint32', 2),
      patchClasses: createTransientView(graph, 'classes', 'uint32', 2),
      width: 2,
      height: 1,
      classCount: 2,
      connectivity: 4 as const,
      cellSize: [1e-47, 1] as const
    };
    const before = inventory(graph);
    expect(() => addLandscapeMetricsRecipe(graph, props)).toThrow(/positive finite float32/);
    expect(inventory(graph)).toEqual(before);
    addLandscapeMetricsRecipe(graph, {...props, cellSize: [1, 1]});
  } finally {
    device.destroy();
  }
});

it('S4 reference preserves recipe and field context for foreign inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const foreign = new GPUCommandGraph(device);
  try {
    expect(() =>
      addBuildingTessellationReferenceRecipe(graph, {
        id: 'review-foreign',
        segments: createTransientView(foreign, 'foreign', 'float32x4', 1),
        segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 1),
        enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
        buildingCount: 1,
        sampleSpacing: 1,
        sampleCapacity: 1,
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 1, nodedSegments: 16, vertices: 8, rings: 2}
      })
    ).toThrow(/review-foreign.*segments/);
  } finally {
    device.destroy();
  }
});

it.each([
  'convex',
  'simple'
] as const)('S4 supplied overlay record retains caller identity after preparation (%s)', enclosureMode => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = {
      enclosureMode,
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      segmentBuildingIds: createTransientView(graph, 'owners', 'uint32', 1),
      enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
      buildingCount: 1,
      sampleSpacing: 1,
      sampleCapacity: 1,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 1, nodedSegments: 16, vertices: 8, rings: 2}
    };
    const provided = addBuildingTessellationReferenceRecipe(graph, {...props, id: 'first'})
      .outputs[0];
    const result = addBuildingTessellationReferenceRecipe(graph, {
      ...props,
      id: 'second',
      output: {overlays: [provided]}
    });
    expect(result.outputs[0]).toBe(provided);
    expect(result.outputs[0].geometry.positions).toBe(provided.geometry.positions);
  } finally {
    device.destroy();
  }
});

it('S3 boundary certification tests the stored infinity word rather than a float infinity', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    addPatchProximityRecipe(graph, {
      mode: 'boundary',
      patches: createTransientView(graph, 'patches', 'uint32', 3),
      patchClasses: createTransientView(graph, 'classes', 'uint32', 2),
      width: 3,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: 0,
      distanceModel: 'cell-center'
    });
    const summary = getGPUCommandNodeKernelSources(
      (graph as unknown as {nodes: unknown[]}).nodes
    ).find(kernel => kernel.id.endsWith('-summary'))!;
    expect(summary.source).toMatch(/nearestWord\s*==\s*0x7f800000u/);
    expect(summary.source).not.toMatch(/nearest\s*==\s*infinity/);
  } finally {
    device.destroy();
  }
});
