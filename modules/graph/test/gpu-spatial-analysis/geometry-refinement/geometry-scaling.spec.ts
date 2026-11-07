// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSubdividePolygons,
  GPUMaximumInscribedCircle,
  GPUVectorStraightSkeleton
} from '../../../src/gpu-spatial-analysis/geometry-refinement';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('measures complete geometry graphs and selective join candidate reduction at three batch scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const measurements = [];
  for (const count of [16, 64, 256]) {
    const buffers: Buffer[] = [];
    let serial = 0;
    const makeBuffer = (values: Float32Array | Uint32Array | number) => {
      const buffer =
        typeof values === 'number'
          ? createOutputBuffer(device, values)
          : createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const coordinates = Float32Array.from(
      Array.from({length: count}, (_, feature) =>
        [0, 0, 8, 0, 8, 2, 2, 2, 2, 8, 0, 8].map((value, index) =>
          index % 2 === 0 ? value + feature * 16 : value
        )
      ).flat()
    );
    const uploadStart = performance.now();
    const positionsBuffer = makeBuffer(coordinates);
    const ringOffsetsBuffer = makeBuffer(
      Uint32Array.from({length: count + 1}, (_, row) => row * 6)
    );
    const parentOffsetsBuffer = makeBuffer(Uint32Array.from({length: count + 1}, (_, row) => row));
    const uploadMilliseconds = performance.now() - uploadStart;
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x4'>(
      format: Format,
      length: number,
      buffer: Buffer
    ) => importGraphBuffer(graph, `geometry-${serial++}`, buffer, format, length);
    const positions = view('float32x2', count * 6, positionsBuffer);
    const ringOffsets = view('uint32', count + 1, ringOffsetsBuffer);
    const polygonOffsets = view('uint32', count + 1, parentOffsetsBuffer);
    const featureOffsets = view('uint32', count + 1, parentOffsetsBuffer);
    const piecePositionsBuffer = makeBuffer(count * 8 * 4 * 2),
      piecesBuffer = makeBuffer(count * 8 * 4),
      subdivisionStatusBuffer = makeBuffer(4),
      circlesBuffer = makeBuffer(count * 4),
      circleStatusBuffer = makeBuffer(count),
      skeletonStatusBuffer = makeBuffer(count * 4);
    const buildStart = performance.now();
    graph.add(
      new GPUSubdividePolygons({
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        method: 'rectilinear',
        maximumVertices: 4,
        output: {
          positions: view('float32x2', count * 8 * 4, piecePositionsBuffer),
          pieces: view('uint32x4', count * 8, piecesBuffer),
          status: view('uint32x4', 1, subdivisionStatusBuffer)
        }
      })
    );
    graph.add(
      new GPUMaximumInscribedCircle({
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        tolerance: 0.01,
        cellCapacity: 256,
        maximumIterations: 2048,
        output: {
          circles: view('float32x4', count, circlesBuffer),
          witnesses: view('float32x2', count, makeBuffer(count * 2)),
          status: view('uint32', count, circleStatusBuffer)
        }
      })
    );
    graph.add(
      new GPUVectorStraightSkeleton({
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        maximumVertices: 6,
        maximumEvents: 16,
        segmentCapacity: 24,
        offsetDistance: 0.25,
        output: {
          segments: view('float32x4', count * 24, makeBuffer(count * 24 * 4)),
          sourceEdges: view('uint32x2', count * 24, makeBuffer(count * 24 * 2)),
          offsetPositions: view('float32x2', count * 38, makeBuffer(count * 38 * 2)),
          offsetLinks: view('uint32x2', count * 38, makeBuffer(count * 38 * 2)),
          status: view('uint32x4', count, skeletonStatusBuffer)
        }
      })
    );
    const constructionMilliseconds = performance.now() - buildStart;
    const compileStart = performance.now();
    const compiled = graph.compile();
    const compilationMilliseconds = performance.now() - compileStart;
    const execute = async (command: typeof compiled) => {
      const start = performance.now();
      submitGraph(device, command);
      await device.handle.queue.onSubmittedWorkDone();
      return performance.now() - start;
    };
    try {
      const times = [];
      for (let run = 0; run < 4; run++) times.push(await execute(compiled));
      const readStart = performance.now();
      const subdivisionStatus = await readUint32(subdivisionStatusBuffer, 4);
      const circleStatus = await readUint32(circleStatusBuffer, count);
      const circles = await readFloat32(circlesBuffer, count * 4);
      const skeletonStatus = await readUint32(skeletonStatusBuffer, count * 4);
      const readbackMilliseconds = performance.now() - readStart;
      expect(subdivisionStatus.slice(2)).toEqual([0, 0]);
      expect(subdivisionStatus[0]).toBeGreaterThan(count);
      expect(circleStatus).toEqual(new Array(count).fill(0));
      for (let row = 0; row < count; row++) {
        expect(circles[row * 4 + 2]).toBeCloseTo(4 - 2 * Math.SQRT2, 2);
        expect(circles[row * 4 + 3]).toBeLessThanOrEqual(0.01);
        expect(skeletonStatus[row * 4 + 3]).toBe(0);
        expect(skeletonStatus[row * 4 + 2]).toBe(6);
      }
      const queries = Float32Array.from(
        Array.from({length: count * 64}, (_, row) => [
          Math.floor(row / 64) * 16 + (row % 8) + 0.37,
          Math.floor((row % 64) / 8) + 0.29
        ]).flat()
      );
      const queriesBuffer = makeBuffer(queries);
      const joinMeasurements = [];
      for (const split of [false, true]) {
        const joinGraph = new GPUCommandGraph(device);
        const queryView = importGraphBuffer(
          joinGraph,
          'queries',
          queriesBuffer,
          'float32x2',
          queries.length / 2
        );
        const countBuffer = makeBuffer(queries.length / 2);
        const result = importGraphBuffer(
          joinGraph,
          'counts',
          countBuffer,
          'uint32',
          queries.length / 2
        );
        const piecePositions = importGraphBuffer(
          joinGraph,
          'piece-positions',
          piecePositionsBuffer,
          'float32x2',
          count * 8 * 4
        );
        const pieces = importGraphBuffer(joinGraph, 'pieces', piecesBuffer, 'uint32x4', count * 8);
        joinGraph.add(
          createWGSLKernelNode(joinGraph, {
            id: 'candidate-join',
            operation: 'GeometryCandidateBenchmark',
            invocationCount: queries.length / 2,
            bindings: [
              {name: 'queries', view: queryView, type: 'f32', access: 'read'},
              {name: 'positions', view: piecePositions, type: 'f32', access: 'read'},
              {name: 'pieces', view: pieces, type: 'u32', access: 'read'},
              {name: 'result', view: result, type: 'u32', access: 'read_write'}
            ],
            body: `let query=vec2f(queries[queriesOffset+index*2u],queries[queriesOffset+index*2u+1u]);var candidates=0u;
for(var row=0u;row<${split ? subdivisionStatus[0] : count}u;row++) {
 var lower=vec2f(f32(row)*16.0,0.0);var upper=lower+vec2f(8.0);
 ${split ? 'lower=vec2f(3e38);upper=vec2f(-3e38);for(var vertex=0u;vertex<pieces[piecesOffset+row*4u];vertex++) {let base=positionsOffset+(row*4u+vertex)*2u;let point=vec2f(positions[base],positions[base+1u]);lower=min(lower,point);upper=max(upper,point);}' : ''}
 if(all(query>=lower)&&all(query<=upper)) {candidates++;}
}
result[resultOffset+index]=candidates;`
          })
        );
        const joinCompiled = joinGraph.compile();
        try {
          const joinTimes = [];
          for (let run = 0; run < 4; run++) joinTimes.push(await execute(joinCompiled));
          const candidateCounts = await readUint32(countBuffer, queries.length / 2);
          joinMeasurements.push({
            split,
            firstMilliseconds: joinTimes[0],
            medianWarmMilliseconds: joinTimes.slice(1).sort((left, right) => left - right)[1],
            candidates: candidateCounts.reduce((sum, value) => sum + value, 0)
          });
        } finally {
          joinCompiled.destroy();
        }
      }
      expect(joinMeasurements[1].candidates).toBeLessThan(joinMeasurements[0].candidates * 0.6);
      const changed = coordinates.map((value, index) => (index % 2 === 1 ? value + 0.001 : value));
      const updateStart = performance.now();
      positionsBuffer.write(changed);
      const updateUploadMilliseconds = performance.now() - updateStart;
      const changedExecutionMilliseconds = await execute(compiled);
      measurements.push({
        features: count,
        inputVertices: count * 6,
        uploadMilliseconds,
        constructionMilliseconds,
        compilationMilliseconds,
        firstExecutionMilliseconds: times[0],
        medianWarmExecutionMilliseconds: times.slice(1).sort((left, right) => left - right)[1],
        readbackMilliseconds,
        updateUploadMilliseconds,
        changedExecutionMilliseconds,
        join: joinMeasurements
      });
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }
  console.log(`GEOMETRY_SCALING ${JSON.stringify({device: device.info, measurements})}`);
});
