// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConvexStraightSkeleton over convex polygons: the skeleton arcs (white) and the inward offset
// ring at `offsetDistance` (orange), both read from the output buffers.

import {GPUConvexStraightSkeleton} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {addWrittenSegments} from './shared/shape-segments';
import {parameters} from './convex-straight-skeleton.parameters';

const VERTEX_COUNT = 6;
const OPTIONS = {irregularity: 0, vertexCount: VERTEX_COUNT};

export default defineOperationRun({
  parameters,
  caption: 'Skeleton arcs and the inward offset of each convex polygon.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {...OPTIONS, stretch: 0.4})
  }),
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const count = inputs.set.count;
    const segmentCapacity = VERTEX_COUNT * 2;
    const segments = builder.allocate('segments', 'float32x4', count * segmentCapacity);
    const offsetPositions = builder.allocate('offset-positions', 'float32x2', count * VERTEX_COUNT);
    builder.graph.add(
      new GPUConvexStraightSkeleton({
        positions: polygons.positions.view,
        ringOffsets: polygons.ringOffsets.view,
        maximumVertices: VERTEX_COUNT,
        segmentCapacity,
        offsetDistance: values.offsetDistance,
        output: {
          segments: segments.view,
          sourceEdges: builder.allocate('source-edges', 'uint32x2', count * segmentCapacity).view,
          offsetPositions: offsetPositions.view,
          status: builder.allocate('status', 'uint32x4', count).view
        }
      })
    );
    return {
      outputs: {
        ...polygons,
        offsetPositions,
        skeleton: addWrittenSegments(builder, 'skeleton', segments),
        offsetEdges: builder.upload(
          'offset-edges',
          'uint32x2',
          Uint32Array.from({length: count * VERTEX_COUNT * 2}, (_, slot) => {
            const edge = Math.floor(slot / 2);
            const base = Math.floor(edge / VERTEX_COUNT) * VERTEX_COUNT;
            return slot % 2 === 0 ? edge : base + ((edge - base + 1) % VERTEX_COUNT);
          })
        )
      },
      update: current =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            ...OPTIONS,
            stretch: current.stretch
          })
        )
    };
  },
  rebuildOn: ['offsetDistance'],
  draw: ({inputs, outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.skeleton, color: [...theme.ink, 1], radius: 1.6},
    {
      layer: 'segments',
      nodes: outputs.offsetPositions,
      edges: outputs.offsetEdges,
      color: [1, 0.6, 0.2, 1],
      radius: 2
    }
  ]
});
