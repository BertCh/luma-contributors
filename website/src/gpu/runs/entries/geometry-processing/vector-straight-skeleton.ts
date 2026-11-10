// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVectorStraightSkeleton over star-shaped polygons with reflex corners: the skeleton arcs
// (white) and the inward offset curve (orange) at `offsetDistance`.

import {GPUVectorStraightSkeleton} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {
  findPointAtDistance,
  getBinIndex,
  getNearestValue,
  getPolygonDepth
} from './shared/pointer-support';
import {addKernel, addWrittenSegments} from './shared/shape-segments';
import {parameters} from './vector-straight-skeleton.parameters';

const MAXIMUM_VERTICES = 16;
const MAXIMUM_EVENTS = 32;
const SEGMENT_CAPACITY = 64;
const OPTIONS = {vertexCount: 12, rows: 1, columns: 3};
/** offsetDistance is a constructor prop: the pointer settles on one of these prebuilt values. */
const OFFSETS = [0.008, 0.02, 0.032, 0.044, 0.06];
/** Distance from the nearest edge that spans the five offset bins. */
const REACH = 0.2;

export default defineOperationRun({
  parameters,
  caption: 'Skeleton arcs and the inward offset curve of star-shaped polygons.',
  buildCache: 5,
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {...OPTIONS, irregularity: 0.4});
    return {
      seed,
      world,
      set,
      restPoint: findPointAtDistance(
        (x, y) => {
          const hit = getPolygonDepth(set, x, y);
          return hit?.isInside ? hit.depth : null;
        },
        0.02,
        world
      )
    };
  },
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const count = inputs.set.count;
    const slots = MAXIMUM_VERTICES + 2 * MAXIMUM_EVENTS;
    const segments = builder.allocate('segments', 'float32x4', count * SEGMENT_CAPACITY);
    const offsetPositions = builder.allocate('offset-positions', 'float32x2', count * slots);
    const offsetLinks = builder.allocate('offset-links', 'uint32x2', count * slots);
    const status = builder.allocate('status', 'uint32x4', count);
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
          status: status.view
        }
      })
    );
    // Links are previous/next vertex ids local to the polygon block; make them global edges.
    const offsetEdges = builder.allocate('offset-edges', 'uint32x2', count * slots);
    // Rows past a polygon's offset vertex count are unwritten zeros: park them off screen.
    const parkedOffsets = builder.allocate('parked-offsets', 'float32x2', count * slots);
    addKernel(builder, {
      id: 'offset-edges',
      invocationCount: count * slots,
      bindings: [
        {name: 'links', buffer: offsetLinks, access: 'read'},
        {name: 'status', buffer: status, access: 'read'},
        {name: 'positions', buffer: offsetPositions, access: 'read'},
        {name: 'edges', buffer: offsetEdges, access: 'read_write'},
        {name: 'parked', buffer: parkedOffsets, access: 'read_write'}
      ],
      body: `let isWritten = index % ${slots}u < status[statusOffset + (index / ${slots}u) * 4u + 2u];
  let nextVertex = links[linksOffset + index * 2u + 1u];
  let hasNext = isWritten && nextVertex != 0xffffffffu;
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = select(index, (index / ${slots}u) * ${slots}u + nextVertex, hasNext);
  parked[parkedOffset + index * 2u] = select(1000000.0, positions[positionsOffset + index * 2u], isWritten);
  parked[parkedOffset + index * 2u + 1u] = select(1000000.0, positions[positionsOffset + index * 2u + 1u], isWritten);`
    });
    return {
      outputs: {
        ...polygons,
        offsetPositions: parkedOffsets,
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
  // The pointer's depth inside a polygon (its distance from the nearest edge outside) is the offsetDistance.
  interaction: {
    kind: 'reach',
    hint: 'go deeper inside',
    rest: ({inputs, toUnit}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, values}) {
      const hit = getPolygonDepth(inputs.set, pointer.x, pointer.y);
      // Inside a polygon the offset curve passes through the pointer; outside, the distance picks a bin.
      const offsetDistance = hit?.isInside
        ? getNearestValue(OFFSETS, hit.depth)
        : OFFSETS[getBinIndex(hit?.depth ?? 0, REACH, OFFSETS.length)];
      return {
        settle: {offsetDistance},
        glyphs: hit
          ? [
              {shape: 'line', from: [pointer.x, pointer.y], to: hit.point, dashed: true},
              {shape: 'dot', at: hit.point}
            ]
          : [],
        readout:
          offsetDistance === values.offsetDistance
            ? `offsetDistance ${offsetDistance.toFixed(3)}`
            : `offsetDistance ${values.offsetDistance.toFixed(3)} -> ${offsetDistance.toFixed(3)}`
      };
    }
  },
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
