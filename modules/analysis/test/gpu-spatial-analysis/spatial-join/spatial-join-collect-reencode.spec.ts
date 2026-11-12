// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPointInPolygonJoin,
  GPU_SPATIAL_JOIN_NO_FEATURE
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  buildPolygonFeatureArrays,
  classifyPointInFeature,
  type OraclePolygonFeature
} from './spatial-join-oracle';

it('join collection resets stored and required counts without overwriting candidate or uncertain counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'join-collect-reencode'});
  const pointCount = 257;
  const matchCapacity = 128;
  const ring: [number, number][] = [
    [0, 0],
    [4, 0],
    [0, 4]
  ];
  const feature: OraclePolygonFeature = [[ring]];
  const arrays = buildPolygonFeatureArrays([feature]);
  const originalPoints: [number, number][] = Array.from({length: pointCount}, (_, row) =>
    row % 3 === 0 ? [0.5, 0.5] : row % 3 === 1 ? [1, 1] : [3, 3]
  );
  // Every point remains within the triangle's bounding box, but outside the triangle itself.
  const unmatchedPoints: [number, number][] = originalPoints.map(() => [3, 3]);
  const sourceIds = Uint32Array.from(originalPoints, (_, row) => 1000 + row);
  const buffers: Buffer[] = [];
  const importInput = <Format extends 'float32x2' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, format, length)};
  };
  const importOutput = (id: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', length)};
  };
  const points = importInput(
    'points',
    Float32Array.from(originalPoints.flat()),
    'float32x2',
    pointCount
  );
  const pointFeatureIds = importOutput('point-feature-ids', pointCount);
  const featureCounts = importOutput('feature-counts', 1);
  const candidateCount = importOutput('candidate-count', 1);
  const uncertainCount = importOutput('uncertain-count', 1);
  const overflow = importOutput('overflow', 1);
  const matchIds = importOutput('match-ids', matchCapacity);
  const matchCount = importOutput('match-count', 1);
  const matchRequiredCount = importOutput('match-required-count', 1);
  const matchOverflow = importOutput('match-overflow', 1);
  graph.add(
    new GPUPointInPolygonJoin({
      points: {
        kind: 'points',
        positions: points.view,
        sourceIds: importInput('source-ids', sourceIds, 'uint32', pointCount).view
      },
      polygons: {
        kind: 'polygons',
        positions: importInput(
          'polygon-positions',
          arrays.polygonPositions,
          'float32x2',
          arrays.polygonPositions.length / 2
        ).view,
        featureOffsets: importInput(
          'feature-offsets',
          arrays.featureOffsets,
          'uint32',
          arrays.featureOffsets.length
        ).view,
        polygonOffsets: importInput(
          'polygon-offsets',
          arrays.polygonOffsets,
          'uint32',
          arrays.polygonOffsets.length
        ).view,
        ringOffsets: importInput(
          'ring-offsets',
          arrays.ringOffsets,
          'uint32',
          arrays.ringOffsets.length
        ).view
      },
      boundaryRule: 'closed',
      candidateCapacity: pointCount,
      output: {
        pointFeatureIds: pointFeatureIds.view,
        featureCounts: featureCounts.view,
        matches: {
          ids: matchIds.view,
          status: {
            count: matchCount.view,
            requiredCount: matchRequiredCount.view,
            overflow: matchOverflow.view
          }
        },
        status: {
          candidateCount: candidateCount.view,
          uncertainCount: uncertainCount.view,
          overflow: overflow.view
        }
      }
    })
  );
  const compiled = graph.compile();
  const minimumX = Math.min(...ring.map(point => point[0]));
  const maximumX = Math.max(...ring.map(point => point[0]));
  const minimumY = Math.min(...ring.map(point => point[1]));
  const maximumY = Math.max(...ring.map(point => point[1]));
  try {
    for (const currentPoints of [originalPoints, unmatchedPoints]) {
      points.buffer.write(Float32Array.from(currentPoints.flat()));
      submitGraph(device, compiled, undefined);
      const classifications = currentPoints.map(point => classifyPointInFeature(point, feature));
      const expectedFeatureIds = classifications.map(classification =>
        classification === 'outside' ? GPU_SPATIAL_JOIN_NO_FEATURE : 0
      );
      const expectedMatches = Array.from(sourceIds).filter(
        (_, row) => expectedFeatureIds[row] !== GPU_SPATIAL_JOIN_NO_FEATURE
      );
      const expectedCandidateCount = currentPoints.filter(
        ([x, y]) => x >= minimumX && x <= maximumX && y >= minimumY && y <= maximumY
      ).length;
      const expectedStoredCount = Math.min(expectedMatches.length, matchCapacity);
      const expectedOverflow = Number(expectedMatches.length > matchCapacity);
      // The dyadic points are strictly inside or outside, so exact refinement has no uncertainty.
      expect(classifications.every(classification => classification !== 'boundary')).toBe(true);
      expect(await readUint32(matchCount.buffer)).toEqual([expectedStoredCount]);
      expect(await readUint32(matchRequiredCount.buffer)).toEqual([expectedMatches.length]);
      expect(await readUint32(matchIds.buffer, expectedStoredCount)).toEqual(
        expectedMatches.slice(0, matchCapacity)
      );
      // Public statuses publish state words 0 and 1; collect must write only state word 2.
      expect(await readUint32(candidateCount.buffer)).toEqual([expectedCandidateCount]);
      expect(await readUint32(uncertainCount.buffer)).toEqual([0]);
      expect(await readUint32(overflow.buffer)).toEqual([expectedOverflow]);
      expect(await readUint32(matchOverflow.buffer)).toEqual([expectedOverflow]);
      expect(await readUint32(pointFeatureIds.buffer)).toEqual(expectedFeatureIds);
      expect(await readUint32(featureCounts.buffer)).toEqual([expectedMatches.length]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
