// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConvexStraightSkeleton over convex polygons: the skeleton arcs (white) and the inward offset
// ring at `offsetDistance` (orange), both read from the output buffers.

import {GPUConvexStraightSkeleton} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {
  findPointAtDistance,
  getBinIndex,
  getNearestValue,
  getPolygonDepth
} from './shared/pointer-support';
import {addKernel} from './shared/shape-segments';
import {parameters} from './convex-straight-skeleton.parameters';

const VERTEX_COUNT = 6;
const OPTIONS = {convexShore: true, vertexCount: VERTEX_COUNT, rows: 2, columns: 4};
/** offsetDistance is a constructor prop: the pointer settles on one of these prebuilt values. */
const OFFSETS = [0.01, 0.02, 0.035, 0.05, 0.07];
/** Distance from the nearest edge that spans the five offset bins. */
const REACH = 0.2;

export default defineOperationRun({
  parameters,
  caption: 'Skeleton arcs and the inward offset of six-vertex Great Lakes hulls.',
  buildCache: 5,
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {...OPTIONS, stretch: 0.2});
    return {
      seed,
      world,
      set,
      restPoint: findPointAtDistance(
        (x, y) => {
          const hit = getPolygonDepth(set, x, y);
          return hit?.isInside ? hit.depth : null;
        },
        0.035,
        world
      )
    };
  },
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const count = inputs.set.count;
    const segmentCapacity = VERTEX_COUNT * 2;
    const segments = builder.allocate('segments', 'float32x4', count * segmentCapacity);
    const offsetPositions = builder.allocate('offset-positions', 'float32x2', count * VERTEX_COUNT);
    const status = builder.allocate('status', 'uint32x4', count);
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
          status: status.view
        }
      })
    );
    const skeleton = builder.allocate('visible-arcs', 'float32x4', segments.length);
    addKernel(builder, {
      id: 'visible-arcs',
      invocationCount: segments.length,
      bindings: [
        {name: 'segments', buffer: segments, access: 'read'},
        {name: 'status', buffer: status, access: 'read'},
        {name: 'visible', buffer: skeleton, access: 'read_write'}
      ],
      body: `let count = status[statusOffset + (index / ${segmentCapacity}u) * 4u];
        for (var component = 0u; component < 4u; component++) {
          visible[visibleOffset + index * 4u + component] = select(-9.0, segments[segmentsOffset + index * 4u + component], index % ${segmentCapacity}u < count);
        }`
    });
    // Rows past an offset ring's corner count are unwritten zeros: park them off screen.
    const parkedOffsets = builder.allocate('parked-offsets', 'float32x2', offsetPositions.length);
    addKernel(builder, {
      id: 'parked-offsets',
      invocationCount: offsetPositions.length,
      bindings: [
        {name: 'status', buffer: status, access: 'read'},
        {name: 'positions', buffer: offsetPositions, access: 'read'},
        {name: 'parked', buffer: parkedOffsets, access: 'read_write'}
      ],
      body: `let isWritten = index % ${VERTEX_COUNT}u < status[statusOffset + (index / ${VERTEX_COUNT}u) * 4u + 2u];
  parked[parkedOffset + index * 2u] = select(1000000.0, positions[positionsOffset + index * 2u], isWritten);
  parked[parkedOffset + index * 2u + 1u] = select(1000000.0, positions[positionsOffset + index * 2u + 1u], isWritten);`
    });
    const offsetEdges = builder.allocate('offset-edges', 'uint32x2', offsetPositions.length);
    addKernel(builder, {
      id: 'offset-edges',
      invocationCount: offsetPositions.length,
      bindings: [
        {name: 'status', buffer: status, access: 'read'},
        {name: 'edges', buffer: offsetEdges, access: 'read_write'}
      ],
      body: `let row = index / ${VERTEX_COUNT}u;
        let count = status[statusOffset + row * 4u + 2u];
        let corner = index % ${VERTEX_COUNT}u;
        let base = row * ${VERTEX_COUNT}u;
        edges[edgesOffset + index * 2u] = select(base, index, corner < count);
        edges[edgesOffset + index * 2u + 1u] = select(base, base + (corner + 1u) % max(1u, count), corner < count);`
    });
    return {
      outputs: {...polygons, offsetPositions: parkedOffsets, skeleton, offsetEdges},
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
