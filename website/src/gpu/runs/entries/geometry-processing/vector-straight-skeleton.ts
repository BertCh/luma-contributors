// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVectorStraightSkeleton over star-shaped polygons with reflex corners: the skeleton arcs
// (white) and the inward offset curve (orange) at `offsetDistance`.

import {GPUVectorStraightSkeleton} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {addKernel, addWrittenSegments} from './shared/shape-segments';
import {parameters} from './vector-straight-skeleton.parameters';

const MAXIMUM_VERTICES = 16;
const MAXIMUM_EVENTS = 32;
const SEGMENT_CAPACITY = 64;
const OPTIONS = {vertexCount: 12, rows: 2, columns: 4};

export default defineOperationRun({
  parameters,
  caption: 'Skeleton arcs and the inward offset curve of star-shaped polygons.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {...OPTIONS, irregularity: 0.4})
  }),
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const count = inputs.set.count;
    const slots = MAXIMUM_VERTICES + 2 * MAXIMUM_EVENTS;
    const segments = builder.allocate('segments', 'float32x4', count * SEGMENT_CAPACITY);
    const offsetPositions = builder.allocate('offset-positions', 'float32x2', count * slots);
    const offsetLinks = builder.allocate('offset-links', 'uint32x2', count * slots);
    builder.graph.add(
      new GPUVectorStraightSkeleton({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        maximumVertices: MAXIMUM_VERTICES,
        maximumEvents: MAXIMUM_EVENTS,
        segmentCapacity: SEGMENT_CAPACITY,
        offsetDistance: values.offsetDistance,
        output: {
          segments: segments.view,
          sourceEdges: builder.allocate('source-edges', 'uint32x2', count * SEGMENT_CAPACITY).view,
          offsetPositions: offsetPositions.view,
          offsetLinks: offsetLinks.view,
          status: builder.allocate('status', 'uint32x4', count).view
        }
      })
    );
    // Links are previous/next vertex ids local to the polygon block; make them global edges.
    const offsetEdges = builder.allocate('offset-edges', 'uint32x2', count * slots);
    addKernel(builder, {
      id: 'offset-edges',
      invocationCount: count * slots,
      bindings: [
        {name: 'links', buffer: offsetLinks, access: 'read'},
        {name: 'edges', buffer: offsetEdges, access: 'read_write'}
      ],
      body: `let nextVertex = links[linksOffset + index * 2u + 1u];
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = select(index, (index / ${slots}u) * ${slots}u + nextVertex, nextVertex != 0xffffffffu);`
    });
    return {
      outputs: {
        ...polygons,
        offsetPositions,
        offsetEdges,
        skeleton: addWrittenSegments(builder, 'skeleton', segments)
      },
      update: current =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            ...OPTIONS,
            irregularity: current.irregularity
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
