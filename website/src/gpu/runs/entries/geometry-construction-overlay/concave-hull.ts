// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConcaveHull keeps the Delaunay triangles whose circumradius is below a limit and writes the
// oriented boundary of what remains (an alpha shape): disconnected parts and holes stay separate.
// A small radius hugs the clusters; a large one approaches the convex hull.

import {
  getDelaunayMaximumTriangleCount,
  GPUConcaveHull
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {addEdgeDrawing} from './shared/edges';

import {HIGHLIGHT_COLOR} from './shared/lines';
import {getCentroid, getReachLevel, getReachRest} from './shared/reach';
import {parameters} from './concave-hull.parameters';

// The circumradius limits the pointer's reach picks among.
const LEVELS = [0.04, 0.09, 0.16, 0.26, 0.4] as const;
const REST_LEVEL = 1;

export default defineOperationRun({
  parameters,
  caption: 'Alpha-shape boundary of scattered points; muted dots are the input.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 260, 3, 0.04);
    return {...points, anchor: getCentroid(points.positions)};
  },
  build(builder, points, values) {
    const pointCount = points.count;
    const capacity = getDelaunayMaximumTriangleCount(pointCount);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const triangles = builder.allocate('triangles', 'uint32x3', capacity);
    const triangleValidity = builder.allocate('triangle-validity', 'uint32', capacity);
    const boundaryEdges = builder.allocate('boundary-edges', 'uint32x2', capacity * 3);
    const edgeValidity = builder.allocate('edge-validity', 'uint32', capacity * 3);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPUConcaveHull({
        positions: positions.view,
        maximumCircumradius: values.maximumCircumradius,
        output: {
          triangles: triangles.view,
          triangleValidity: triangleValidity.view,
          boundaryEdges: boundaryEdges.view,
          edgeValidity: edgeValidity.view,
          status: {converged: converged.view}
        }
      })
    );
    const drawing = addEdgeDrawing(builder, 'hull', positions, boundaryEdges, edgeValidity);
    return {
      outputs: {positions, ...drawing}
    };
  },
  rebuildOn: ['maximumCircumradius'],
  buildCache: 5,
  // The distance from the cloud's centroid to the pointer picks the circumradius limit among five
  // values (applied once it rests); the dashed ball at the cursor is that limit.
  interaction: {
    kind: 'reach',
    hint: 'reach out to loosen the hull',
    rest: ({inputs, toUnit}) => toUnit(getReachRest(inputs.anchor, REST_LEVEL, LEVELS.length)),
    resolve(pointer, {inputs}) {
      const level = getReachLevel(inputs.anchor, [pointer.x, pointer.y], LEVELS.length);
      return {
        settle: {maximumCircumradius: LEVELS[level]},
        glyphs: [
          {shape: 'dot', at: inputs.anchor},
          {shape: 'line', from: inputs.anchor, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'ring', at: [pointer.x, pointer.y], radius: LEVELS[level], dashed: true}
        ],
        readout: `maximumCircumradius ${LEVELS[level].toFixed(2)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3.5},
    {layer: 'lines', segments: outputs.segments, color: HIGHLIGHT_COLOR, radius: 2.4}
  ]
});
