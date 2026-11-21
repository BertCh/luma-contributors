// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared by the edge bundling correctness spec and its timing benchmark
// (`gpu-edge-bundling-bench.spec.ts`).

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUEdgeBundling,
  getGPUEdgeBundlingParameterValues,
  type GPUEdgeBundlingSettings
} from '../../../src/gpu-network/edge-bundling/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

export type Scene = {
  positions: Float32Array;
  sources: Uint32Array;
  targets: Uint32Array;
  mask?: Uint32Array;
};

export type FixtureOptions = {
  pointsPerEdge?: number;
  iterations?: number;
  densityResolution?: number;
  parameterFormat?: 'uint32' | 'float32';
  parameters?: GPUEdgeBundlingSettings;
  withIndices?: boolean;
  geographic?: boolean;
};

export type Fixture = {
  edgeCount: number;
  pointsPerEdge: number;
  iterations: number;
  densityResolution: number;
  compiled: CompiledGPUCommandGraph;
  nodeCount: number;
  buffers: Buffer[];
  parameterBuffer?: GPUParameterBuffer;
  parameterFormat: 'uint32' | 'float32';
  pathsBuffer: Buffer;
  startIndicesBuffer?: Buffer;
  drawRecordBuffer?: Buffer;
  setParameters(values: GPUEdgeBundlingSettings): void;
  run(): Promise<Float32Array>;
  destroy(): void;
};

export function createFixture(device: Device, scene: Scene, options: FixtureOptions = {}): Fixture {
  const pointsPerEdge = options.pointsPerEdge ?? 16;
  const iterations = options.iterations ?? 15;
  const densityResolution = options.densityResolution ?? 256;
  const edgeCount = scene.sources.length;
  const vertexCount = scene.positions.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'edge-bundling'});
  const buffers = [
    createInputBuffer(device, scene.positions),
    createInputBuffer(device, scene.sources),
    createInputBuffer(device, scene.targets)
  ];
  const maskBuffer = scene.mask ? createInputBuffer(device, scene.mask) : undefined;
  const pathsBuffer = createOutputBuffer(device, edgeCount * pointsPerEdge * 2);
  const startIndicesBuffer = options.withIndices
    ? createOutputBuffer(device, edgeCount + 1)
    : undefined;
  const drawRecordBuffer = options.withIndices ? createOutputBuffer(device, 4) : undefined;
  buffers.push(pathsBuffer);
  if (maskBuffer) buffers.push(maskBuffer);
  if (startIndicesBuffer) buffers.push(startIndicesBuffer);
  if (drawRecordBuffer) buffers.push(drawRecordBuffer);
  const parameterFormat = options.parameterFormat ?? 'uint32';
  const parameterBuffer = options.parameters
    ? new GPUParameterBuffer(device, {
        id: 'bundling-parameters',
        format: parameterFormat,
        length: 5,
        values:
          parameterFormat === 'uint32'
            ? getGPUEdgeBundlingParameterValues(options.parameters, 'uint32')
            : getGPUEdgeBundlingParameterValues(options.parameters, 'float32')
      })
    : undefined;
  const contributor = new GPUEdgeBundling({
    nodePositions: importGraphBuffer(graph, 'positions', buffers[0], 'float32x2', vertexCount),
    sourceNodes: importGraphBuffer(graph, 'sources', buffers[1], 'uint32', edgeCount),
    targetNodes: importGraphBuffer(graph, 'targets', buffers[2], 'uint32', edgeCount),
    edgeMask: maskBuffer && importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', edgeCount),
    pointsPerEdge,
    maximumIterations: iterations,
    densityResolution,
    spatialContext: options.geographic
      ? {coordinateSpace: 'longitude-latitude', metric: 'none'}
      : undefined,
    parameters: parameterBuffer?.importToGraph(graph),
    output: {
      paths: importGraphBuffer(graph, 'paths', pathsBuffer, 'float32x2', edgeCount * pointsPerEdge),
      lineOffsets:
        startIndicesBuffer &&
        importGraphBuffer(graph, 'start-indices', startIndicesBuffer, 'uint32', edgeCount + 1),
      drawRecord:
        drawRecordBuffer && importGraphBuffer(graph, 'draw-record', drawRecordBuffer, 'uint32', 4)
    }
  });
  graph.add(contributor);
  const nodeCount =
    4 + 3 * iterations + (options.withIndices ? 1 : 0) + (options.parameters ? 1 : 0);
  const compiled = graph.compile();
  return {
    edgeCount,
    pointsPerEdge,
    iterations,
    densityResolution,
    compiled,
    nodeCount,
    buffers,
    parameterBuffer,
    parameterFormat,
    pathsBuffer,
    startIndicesBuffer,
    drawRecordBuffer,
    setParameters(values) {
      parameterBuffer!.write(
        parameterFormat === 'uint32'
          ? getGPUEdgeBundlingParameterValues(values, 'uint32')
          : getGPUEdgeBundlingParameterValues(values, 'float32')
      );
    },
    async run() {
      submitGraph(device, compiled, undefined);
      return Float32Array.from(await readFloat32(pathsBuffer, edgeCount * pointsPerEdge * 2));
    },
    destroy() {
      compiled.destroy();
      parameterBuffer?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

/** Random flows between a few clusters, so bundling has real structure to find. */
export function createClusteredScene(vertexCount: number, edgeCount: number, seed: number): Scene {
  const random = createRandom(seed);
  const centers = [
    [20, 25],
    [80, 30],
    [30, 80],
    [85, 85]
  ];
  const positions = new Float32Array(vertexCount * 2);
  for (let v = 0; v < vertexCount; v++) {
    const center = centers[v % centers.length];
    positions[v * 2] = center[0] + (random() - 0.5) * 12;
    positions[v * 2 + 1] = center[1] + (random() - 0.5) * 12;
  }
  const sources = new Uint32Array(edgeCount);
  const targets = new Uint32Array(edgeCount);
  for (let e = 0; e < edgeCount; e++) {
    sources[e] = Math.floor(random() * vertexCount);
    do {
      targets[e] = Math.floor(random() * vertexCount);
    } while (targets[e] === sources[e]);
  }
  return {positions, sources, targets};
}
