// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {it} from 'vitest';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';
import {createTopologyOutput} from '../geometry-refinement/topology-output';
import {
  expectSharedSourceFeatures,
  withTestResourceCleanup
} from '../geometry-refinement/geometry-output-review-test-utils';

it('make-valid keeps overlapping feature rows separate when both carry one source ID', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const resources: {destroy(): void}[] = [];
  await withTestResourceCleanup(resources, async () => {
    const graph = new GPUCommandGraph(device);
    const inputBuffers: Buffer[] = [];
    // The segment harness registers all allocated inputs, including an error during setup.
    resources.push({destroy: () => inputBuffers.forEach(buffer => buffer.destroy())});
    const polygons = createSegmentGeometry(
      device,
      graph,
      'shared-source',
      [
        {
          kind: 'polygons',
          polygons: [
            [
              [
                [0, 0],
                [4, 0],
                [4, 4],
                [0, 4]
              ]
            ]
          ]
        },
        {
          kind: 'polygons',
          polygons: [
            [
              [
                [2, 1],
                [5, 1],
                [5, 4],
                [2, 4]
              ]
            ]
          ]
        }
      ],
      inputBuffers
    );
    if (polygons.kind !== 'polygons') throw new Error('expected polygons');
    const identities = createInputBuffer(device, Uint32Array.of(77, 77));
    resources.push(identities);
    const outputBuffers = new Map<GraphDataView, Buffer>();
    let serial = 0;
    const output = createTopologyOutput(
      <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
        format: Format,
        length: number
      ) => {
        const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
        const buffer = createOutputBuffer(device, components * length);
        resources.push(buffer);
        const view = importGraphBuffer(graph, `shared-output-${serial++}`, buffer, format, length);
        outputBuffers.set(view, buffer);
        return view;
      }
    );
    const bufferFor = (view: GraphDataView): Buffer => {
      const buffer = outputBuffers.get(view);
      if (!buffer) throw new Error('missing readback buffer');
      return buffer;
    };
    graph.add(
      new GPUMakeValid({
        polygons: {
          ...polygons,
          sourceIds: importGraphBuffer(graph, 'identities', identities, 'uint32', 2)
        },
        intersectionCapacity: 64,
        precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
        output
      })
    );
    const compiled = graph.compile();
    resources.push(compiled);
    submitGraph(device, compiled, undefined);
    const count = (await readUint32(bufferFor(output.status.count), 1))[0];
    const featureOffsets = await readUint32(bufferFor(output.polygons.featureOffsets), count + 1);
    const polygonCount = featureOffsets[count];
    const polygonOffsets = await readUint32(
      bufferFor(output.polygons.polygonOffsets),
      polygonCount + 1
    );
    const ringCount = polygonOffsets[polygonCount];
    const ringOffsets = await readUint32(bufferFor(output.polygons.ringOffsets), ringCount + 1);
    const coordinates = await readFloat32(
      bufferFor(output.polygons.positions),
      2 * ringOffsets[ringCount]
    );
    const publishedPolygons = Array.from({length: polygonCount}, (_, polygon) =>
      Array.from(
        {length: polygonOffsets[polygon + 1] - polygonOffsets[polygon]},
        (_, relativeRing) => {
          const ring = polygonOffsets[polygon] + relativeRing;
          return Array.from(
            {length: ringOffsets[ring + 1] - ringOffsets[ring]},
            (_, relativeVertex) => {
              const vertex = ringOffsets[ring] + relativeVertex;
              return [coordinates[2 * vertex], coordinates[2 * vertex + 1]];
            }
          );
        }
      )
    );
    // Each input is already a valid square: individual areas 4*4=16 and 3*3=9.
    // Feature range [0,2,2] must fail even if polygons and shared provenance look correct.
    expectSharedSourceFeatures({
      featureOffsets,
      polygons: publishedPolygons,
      count,
      sourceIds: await readUint32(bufferFor(output.polygons.sourceIds), count),
      overflow: (await readUint32(bufferFor(output.status.overflow), 1))[0],
      candidateOverflow: (await readUint32(bufferFor(output.status.candidateOverflow), 1))[0]
    });
  });
});
