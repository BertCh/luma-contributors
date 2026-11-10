// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURatioConcaveHull wraps a sampled set of earthquake epicenters. A ratio of 1 is the convex hull; smaller
// ratios erode long Delaunay edges until the boundary hugs the shape. The bright segments are the
// valid boundary edges the operation wrote.

import {
  getDelaunayMaximumTriangleCount,
  GPURatioConcaveHull
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getEarthquakes} from '../../../data/point-observations';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createStaleGuard, getBinCenter, getBinIndex, readFresh} from './shared/pointer-support';
import {addKernel} from './shared/shape-segments';
import {parameters} from './ratio-concave-hull.parameters';

const FAR_AWAY = 1e6;
/** ratio is a constructor prop: the pointer settles on one of these prebuilt values. */
const RATIOS = [0.06, 0.12, 0.2, 0.4, 1];

export default defineOperationRun({
  parameters,
  caption: 'The boundary of a sample of earthquake epicenters, tightened by the edge-length ratio.',
  buildCache: 5,
  createInputs: ({seed, world}) => ({seed, world, guard: createStaleGuard()}),
  build(builder, inputs, values) {
    const random = makeRandom(inputs.seed);
    const {positions} = getEarthquakes(random, inputs.world, values.eventCount);
    const points = builder.upload('points', 'float32x2', positions);
    const capacity = getDelaunayMaximumTriangleCount(values.eventCount);
    const boundaryEdges = builder.allocate('boundary-edges', 'uint32x2', capacity * 3);
    const edgeValidity = builder.allocate('edge-validity', 'uint32', capacity * 3);
    builder.graph.add(
      new GPURatioConcaveHull({
        positions: points.view,
        ratio: values.ratio,
        output: {
          triangles: builder.allocate('triangles', 'uint32x3', capacity).view,
          triangleValidity: builder.allocate('triangle-validity', 'uint32', capacity).view,
          boundaryEdges: boundaryEdges.view,
          edgeValidity: edgeValidity.view,
          status: {converged: builder.allocate('converged', 'uint32', 1).view}
        }
      })
    );
    // Boundary edges as segments; edges the operation rejected are parked outside the view.
    const hull = builder.allocate('hull', 'float32x4', capacity * 3);
    addKernel(builder, {
      id: 'hull-segments',
      invocationCount: capacity * 3,
      bindings: [
        {name: 'points', buffer: points, access: 'read'},
        {name: 'edges', buffer: boundaryEdges, access: 'read'},
        {name: 'validity', buffer: edgeValidity, access: 'read'},
        {name: 'hull', buffer: hull, access: 'read_write'}
      ],
      body: `let isBoundary = validity[validityOffset + index] != 0u;
  let startNode = edges[edgesOffset + index * 2u];
  let endNode = edges[edgesOffset + index * 2u + 1u];
  let base = hullOffset + index * 4u;
  hull[base] = select(${FAR_AWAY}, points[pointsOffset + startNode * 2u], isBoundary);
  hull[base + 1u] = select(${FAR_AWAY}, points[pointsOffset + startNode * 2u + 1u], isBoundary);
  hull[base + 2u] = select(${FAR_AWAY}, points[pointsOffset + endNode * 2u], isBoundary);
  hull[base + 3u] = select(${FAR_AWAY}, points[pointsOffset + endNode * 2u + 1u], isBoundary);`
    });
    return {outputs: {points, hull}};
  },
  // The pointer's x moves along a scale of edge-length ratios; the hull tightens around the sample.
  interaction: {
    kind: 'level',
    hint: 'tighten the hull',
    rest: [getBinCenter(RATIOS.indexOf(0.2), 1, RATIOS.length), 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const index = getBinIndex(pointer.unitX, 1, RATIOS.length);
      const ratio = RATIOS[index];
      const hull = readFresh(inputs.guard, String(values.ratio), peek(outputs.hull));
      let edgeCount = 0;
      if (hull)
        for (let row = 0; row < hull.length; row += 4) if (hull[row] < FAR_AWAY / 2) edgeCount++;
      const railY = 0.05;
      const [left, right] = [world.width * 0.1, world.width * 0.9];
      const glyphs: RunGlyph[] = [{shape: 'line', from: [left, railY], to: [right, railY]}];
      RATIOS.forEach((_, tick) => {
        const x = left + (getBinCenter(tick, 1, RATIOS.length) / 1) * (right - left);
        glyphs.push({shape: 'line', from: [x, railY - 0.015], to: [x, railY + 0.015]});
      });
      glyphs.push({
        shape: 'dot',
        at: [left + getBinCenter(index, 1, RATIOS.length) * (right - left), railY]
      });
      return {
        settle: {ratio},
        glyphs,
        readout:
          ratio === values.ratio
            ? `ratio ${ratio.toFixed(2)} · ${hull ? edgeCount : '...'} boundary edges`
            : `ratio ${values.ratio.toFixed(2)} -> ${ratio.toFixed(2)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'lines', segments: outputs.hull, color: [1, 0.6, 0.2, 1], radius: 2.4}
  ]
});
