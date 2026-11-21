// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createIsochronesTriangleFacilityNodes} from '../../../src/gpu-network/network-isochrones/network-isochrones-passes';
import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

const NONE = 0xffffffff;

it.each([
  1,
  2 ** 50
])('R8/R9 searches the whole labelled raster in world units (scale=%s)', scale => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const triangles = createTransientView(graph, 'triangles', 'float32x2', 9);
  const triangleCount = createTransientView(graph, 'count', 'uint32', 1);
  const parameters = createTransientView(graph, 'parameters', 'float32', 12);
  const pixelFacilities = createTransientView(graph, 'pixels', 'uint32', 6);
  const triangleFacilities = createTransientView(graph, 'facilities', 'uint32', 3);
  try {
    memory.write(
      triangles,
      Float32Array.from(
        [4, 1.5, 6, 1.5, 6, 2.5, 4, 1.5, 6, 2.5, 4, 2.5, 0, 0, 0, 0, 0, 0],
        value => value * scale
      )
    );
    memory.write(triangleCount, Uint32Array.of(2));
    memory.write(
      parameters,
      getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 8 * scale, 3 * scale]})
    );
    memory.write(pixelFacilities, Uint32Array.of(2, 5, 2, NONE, 2, NONE));
    memory.write(triangleFacilities, Uint32Array.of(99, 99, 99));
    const nodes = createIsochronesTriangleFacilityNodes(graph, {
      id: 'nearest',
      width: 2,
      height: 3,
      triangles,
      triangleCount,
      parameters,
      pixelFacilities,
      triangleFacilities
    });
    executeCommandNodesOnCPU(nodes, memory);
    // Hand-worked centroid distances squared: (16/3,11/6) is 20/9 from label 5,
    // (14/3,13/6) is 41/9 from label 5 versus 65/9 from the nearest label 2.
    // Label 5 lies outside both upper-cell triangles' four raster corners.
    // At scale 2**50 even the closest squared distance, (20/9)*2**100, exceeds 1e30.
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual([5, 5, NONE]);
    memory.write(pixelFacilities, new Uint32Array(6).fill(NONE));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual([NONE, NONE, NONE]);
    // The degenerate centroid (4,.5) is equidistant from samples (2,.5) and (6,.5).
    memory.write(triangles, Float32Array.from(new Array(3).fill([4 * scale, 0.5 * scale]).flat()));
    memory.write(triangleCount, Uint32Array.of(1));
    memory.write(pixelFacilities, Uint32Array.of(5, 2, NONE, NONE, NONE, NONE));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual([2, NONE, NONE]);
  } finally {
    device.destroy();
  }
});

it.each([
  0, 5000001
])('B1 compares fan triangles without rounding a world-space centroid (minX=%s)', minimumX => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const triangles = createTransientView(graph, 'fan', 'float32x2', 6);
  const triangleCount = createTransientView(graph, 'count', 'uint32', 1);
  const parameters = createTransientView(graph, 'params', 'float32', 12);
  const pixelFacilities = createTransientView(graph, 'labels', 'uint32', 4);
  const triangleFacilities = createTransientView(graph, 'result', 'uint32', 2);
  try {
    memory.write(
      triangles,
      Float32Array.of(
        minimumX + 0.5,
        0.5,
        minimumX + 1.5,
        0.5,
        minimumX + 1.5,
        1.5,
        minimumX + 0.5,
        0.5,
        minimumX + 1.5,
        1.5,
        minimumX + 0.5,
        1.5
      )
    );
    memory.write(triangleCount, Uint32Array.of(2));
    memory.write(
      parameters,
      getGPUNetworkIsochronesParameterValues({
        breakCount: 1,
        extent: [minimumX, 0, minimumX + 2, 2]
      })
    );
    memory.write(
      pixelFacilities,
      Uint32Array.from(minimumX === 0 ? [9, NONE, NONE, 2] : [1, 9, 1, 9])
    );
    executeCommandNodesOnCPU(
      createIsochronesTriangleFacilityNodes(graph, {
        id: 'fan-label',
        width: 2,
        height: 2,
        triangles,
        triangleCount,
        parameters,
        pixelFacilities,
        triangleFacilities
      }),
      memory
    );
    // Both diagonal-label distances are 5/9; with vertical stripes the right/left
    // fan centroids are strictly nearer the right/left column. Translation changes neither.
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual(
      minimumX === 0 ? [2, 2] : [9, 1]
    );
  } finally {
    device.destroy();
  }
});

it('S1 prunes dense distant samples instead of visiting the whole raster', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const triangles = createTransientView(graph, 'triangles', 'float32x2', 3);
  const triangleCount = createTransientView(graph, 'count', 'uint32', 1);
  const parameters = createTransientView(graph, 'params', 'float32', 12);
  const pixelFacilities = createTransientView(graph, 'labels', 'uint32', 1024);
  const triangleFacilities = createTransientView(graph, 'out', 'uint32', 1);
  const visitedSamples = createTransientView(graph, 'visited', 'uint32', 1);
  try {
    memory.write(triangles, Float32Array.of(0.5, 0.5, 1.5, 0.5, 1.5, 1.5));
    memory.write(triangleCount, Uint32Array.of(1));
    memory.write(
      parameters,
      getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 32, 32]})
    );
    memory.write(pixelFacilities, new Uint32Array(1024).fill(7));
    memory.write(visitedSamples, Uint32Array.of(999));
    const props = {
      id: 'pruned',
      width: 32,
      height: 32,
      triangles,
      triangleCount,
      parameters,
      pixelFacilities,
      triangleFacilities,
      visitedSamples
    };
    const nodes = createIsochronesTriangleFacilityNodes(graph, props);
    for (const {source, constants} of getParsedKernelSources(nodes).configurations) {
      expect(auditWGSLModule(source, constants)).toEqual([]);
    }
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual([7]);
    expect(memory.readUint32(visitedSamples)[0]).toBeGreaterThan(0);
    expect(memory.readUint32(visitedSamples)[0]).toBeLessThan(16);
    memory.write(pixelFacilities, new Uint32Array(1024).fill(NONE));
    executeCommandNodesOnCPU(
      createIsochronesTriangleFacilityNodes(graph, {...props, id: 'empty'}),
      memory
    );
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual([NONE]);
    expect(Array.from(memory.readUint32(visitedSamples))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it('S1 spatial pruning agrees with exact integer distances on an irregular sparse grid', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const vertices: number[] = [];
  for (let row = 0; row < 4; row++)
    for (let column = 0; column < 6; column++)
      vertices.push(
        column + 0.5,
        row + 0.5,
        column + 1.5,
        row + 0.5,
        column + 1.5,
        row + 1.5,
        column + 0.5,
        row + 0.5,
        column + 1.5,
        row + 1.5,
        column + 0.5,
        row + 1.5
      );
  const triangles = createTransientView(
    graph,
    'irregular-triangles',
    'float32x2',
    vertices.length / 2
  );
  const triangleCount = createTransientView(graph, 'irregular-count', 'uint32', 1);
  const parameters = createTransientView(graph, 'irregular-params', 'float32', 12);
  const pixelFacilities = createTransientView(graph, 'irregular-pixels', 'uint32', 35);
  const triangleFacilities = createTransientView(graph, 'irregular-out', 'uint32', 48);
  const labels = new Uint32Array(35).fill(NONE);
  labels[1] = 9;
  labels[17] = 4;
  labels[28] = 2;
  labels[34] = 1;
  // Independent integer arithmetic: centroid displacement times three is
  // (3*(column-sampleColumn)+2, 3*(row-sampleRow)+1), or (+1,+2) for the other fan.
  // Literal reference: scratch/irregular-oracle.py exhaustively enumerates integer distances.
  const expected = [
    9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 2, 2, 4, 4, 4, 4, 4, 4,
    4, 4, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 1, 1, 1, 1
  ];
  try {
    memory.write(triangles, Float32Array.from(vertices));
    memory.write(triangleCount, Uint32Array.of(48));
    memory.write(
      parameters,
      getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 7, 5]})
    );
    memory.write(pixelFacilities, labels);
    const nodes = createIsochronesTriangleFacilityNodes(graph, {
      id: 'irregular',
      width: 7,
      height: 5,
      triangles,
      triangleCount,
      parameters,
      pixelFacilities,
      triangleFacilities
    });
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(triangleFacilities))).toEqual(expected);
  } finally {
    device.destroy();
  }
});
