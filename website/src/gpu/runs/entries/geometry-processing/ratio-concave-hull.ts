// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURatioConcaveHull wraps a C-shaped point cloud. A ratio of 1 is the convex hull; smaller
// ratios erode long Delaunay edges until the boundary hugs the shape. The bright segments are the
// valid boundary edges the operation wrote.

import {
  getDelaunayMaximumTriangleCount,
  GPURatioConcaveHull
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {addKernel} from './shared/shape-segments';
import {parameters} from './ratio-concave-hull.parameters';

const POINT_COUNT = 360;
const FAR_AWAY = 1e6;

export default defineOperationRun({
  parameters,
  caption: 'The boundary of a C-shaped cloud, tightened by the edge-length ratio.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, inputs, values) {
    const random = makeRandom(inputs.seed);
    const positions = new Float32Array(POINT_COUNT * 2);
    for (let point = 0; point < POINT_COUNT; point++) {
      const angle = (values.opening + (2 - 2 * values.opening) * random()) * Math.PI;
      const radius = 0.2 + 0.17 * Math.sqrt(random());
      positions[point * 2] = inputs.world.width * 0.5 + Math.cos(angle) * radius * 1.1;
      positions[point * 2 + 1] = 0.5 + Math.sin(angle) * radius * 1.3;
    }
    const points = builder.upload('points', 'float32x2', positions);
    const capacity = getDelaunayMaximumTriangleCount(POINT_COUNT);
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
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'lines', segments: outputs.hull, color: [1, 0.6, 0.2, 1], radius: 2.4}
  ]
});
