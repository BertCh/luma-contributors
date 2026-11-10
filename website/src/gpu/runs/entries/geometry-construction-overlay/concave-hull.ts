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

import {createPointCarry, getCarriedPositions, updatePointCarry} from './shared/carry';
import {HIGHLIGHT_COLOR} from './shared/lines';
import {parameters} from './concave-hull.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Alpha-shape boundary of scattered points; muted dots are the input.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 260, 3, 0.04);
    return {...points, carry: createPointCarry(points.positions)};
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
      outputs: {positions, ...drawing},
      update: () => positions.write(getCarriedPositions(points.positions, points.carry))
    };
  },
  rebuildOn: ['maximumCircumradius'],
  // The pointer carries the nearest point and the hull re-forms around it; the dashed ball at the
  // cursor is the circumradius limit that decides which triangles stay.
  interaction: {
    kind: 'drag',
    hint: 'carry a point',
    rest: ({inputs, toUnit}) => toUnit([inputs.positions[0], inputs.positions[1]]),
    resolve(pointer, {inputs, values}) {
      const carried = updatePointCarry(inputs.carry, inputs.positions, [pointer.x, pointer.y]);
      return {
        rerun: carried.changed,
        glyphs: [
          {shape: 'ring', at: carried.at, radius: values.maximumCircumradius, dashed: true},
          {shape: 'dot', at: carried.at}
        ],
        readout: `maximumCircumradius ${values.maximumCircumradius.toFixed(2)} · point ${carried.row} carried`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3.5},
    {layer: 'lines', segments: outputs.segments, color: HIGHLIGHT_COLOR, radius: 2.4}
  ]
});
