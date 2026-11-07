// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared body of the two polygon triangulation runs (ear clipping and constrained Delaunay): the
// same star-shaped polygons, with the triangles the operation wrote filled by polygon.

import {
  getPolygonTriangulationIndexCount,
  GPUConstrainedDelaunayTriangulation,
  GPUPolygonTriangulation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../../define-run';
import type {GpuSceneParameter} from '../../../../types';
import {addKernelPass} from './kernel';
import {OUTPUT_COLOR} from './lines';
import {
  createBlobSet,
  createRingSegments,
  getPolygonFields,
  uploadPolygonSet,
  type PolygonSet
} from './polygons';

export function createSeededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Owner polygon of every output triangle, found on the GPU from the polygon triangle offsets. */
function addTriangleOwners(
  builder: RunBuilder,
  triangleOffsets: RunBuffer<'uint32'>,
  triangleCapacity: number
): RunBuffer<'uint32'> {
  const owners = builder.allocate('triangle-owners', 'uint32', triangleCapacity);
  const polygonCount = triangleOffsets.length - 1;
  addKernelPass(builder, {
    id: 'triangle-owner-pass',
    invocationCount: triangleCapacity,
    bindings: [
      {name: 'triangleOffsets', buffer: triangleOffsets, access: 'read'},
      {name: 'owners', buffer: owners, access: 'read_write'}
    ],
    body: /* wgsl */ `
  var low = 0u;
  var high = ${polygonCount + 1}u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (triangleOffsets[triangleOffsetsOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
  }
  owners[ownersOffset + index] = max(low, 1u) - 1u;`
  });
  return owners;
}

export function createTriangulationRun(options: {
  parameters: readonly GpuSceneParameter[];
  constrained: boolean;
  caption: string;
}) {
  return defineOperationRun({
    parameters: options.parameters,
    caption: options.caption,
    createInputs: ({world}) => {
      const getSet = (irregularity: number, hasHoles: boolean): PolygonSet =>
        createBlobSet(createSeededRandom(77), world, {
          rows: 2,
          columns: 3,
          vertexCount: 18,
          irregularity,
          fill: 0.82,
          hasHoles
        });
      return {getSet};
    },
    build(builder, inputs, values) {
      const set = inputs.getSet(values.irregularity, values.hasHoles === 1);
      const polygons = uploadPolygonSet(builder, 'polygons', set);
      const polygonCount = set.polygonOffsets.length - 1;
      const ringCount = set.ringOffsets.length - 1;
      const triangleCapacity =
        getPolygonTriangulationIndexCount(set.positions.length / 2, ringCount, polygonCount) / 3;
      const indices = builder.allocate('indices', 'uint32', triangleCapacity * 3);
      const valid = builder.allocate('valid', 'uint32', polygonCount);
      const triangleOffsets = builder.allocate('triangle-offsets', 'uint32', polygonCount + 1);
      const triangleCounts = builder.allocate('triangle-counts', 'uint32', polygonCount);
      const converged = builder.allocate('converged', 'uint32', polygonCount);
      const output = {
        indices: indices.view,
        valid: valid.view,
        triangleCount: triangleCounts.view,
        triangleOffsets: triangleOffsets.view
      };
      builder.graph.add(
        options.constrained
          ? new GPUConstrainedDelaunayTriangulation({
              ...getPolygonFields(polygons),
              maximumFlips: values.maximumFlips,
              output: {...output, status: {converged: converged.view}}
            })
          : new GPUPolygonTriangulation({
              ...getPolygonFields(polygons),
              largePolygonMethod: 'monotone',
              output
            })
      );
      const owners = addTriangleOwners(builder, triangleOffsets, triangleCapacity);
      const polygonIds = builder.upload(
        'polygon-ids',
        'uint32',
        Uint32Array.from({length: polygonCount}, (_, index) => index)
      );
      const outline = builder.allocate('outline', 'float32x4', createRingSegments(set).length / 4);
      return {
        outputs: {
          positions: polygons.positions,
          indices,
          owners,
          polygonIds,
          outline,
          triangleCapacity
        },
        update: current => {
          const next = inputs.getSet(current.irregularity, values.hasHoles === 1);
          polygons.positions.write(next.positions);
          outline.write(createRingSegments(next));
        }
      };
    },
    rebuildOn: ['hasHoles', 'maximumFlips'],
    draw: ({outputs}) => [
      {
        layer: 'polygons',
        positions: outputs.positions,
        indices: outputs.indices,
        owners: outputs.owners,
        values: outputs.polygonIds,
        triangleCount: outputs.triangleCapacity,
        alpha: 0.85
      },
      {
        layer: 'triangles',
        positions: outputs.positions,
        triangles: outputs.indices,
        count: outputs.triangleCapacity,
        color: [1, 1, 1, 0.55],
        radius: 0.7
      },
      {layer: 'lines', segments: outputs.outline, color: OUTPUT_COLOR, radius: 1.5}
    ]
  });
}
