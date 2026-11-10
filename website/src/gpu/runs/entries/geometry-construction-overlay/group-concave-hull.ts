// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGroupConcaveHull builds one alpha-shape boundary per group label in a single pass. Points are
// muted; each group's boundary has its own color. Labels at or above `groupCount` are ignored.

import {
  getDelaunayMaximumTriangleCount,
  GPUGroupConcaveHull
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {createPointCarry, getCarriedPositions, updatePointCarry} from './shared/carry';
import {addEdgeDrawing} from './shared/edges';
import {parameters} from './group-concave-hull.parameters';

export default defineOperationRun({
  parameters,
  caption: 'One concave hull per label; each boundary has its group color.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 300, 3, 0.08);
    return {...points, carry: createPointCarry(points.positions)};
  },
  build(builder, points, values) {
    const groupCount = values.groupCount;
    const groupCapacity = getDelaunayMaximumTriangleCount(points.count);
    const capacity = groupCapacity * groupCount;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const groupIds = builder.upload('group-ids', 'uint32', points.labels);
    const triangles = builder.allocate('triangles', 'uint32x3', capacity);
    const triangleValidity = builder.allocate('triangle-validity', 'uint32', capacity);
    const boundaryEdges = builder.allocate('boundary-edges', 'uint32x2', capacity * 3);
    const edgeValidity = builder.allocate('edge-validity', 'uint32', capacity * 3);
    const converged = builder.allocate('converged', 'uint32', groupCount);
    builder.graph.add(
      new GPUGroupConcaveHull({
        positions: positions.view,
        groupIds: groupIds.view,
        groupCount,
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
    const drawing = addEdgeDrawing(
      builder,
      'hull',
      positions,
      boundaryEdges,
      edgeValidity,
      groupCapacity * 3
    );
    return {
      outputs: {positions, groupIds, ...drawing},
      update: () => positions.write(getCarriedPositions(points.positions, points.carry))
    };
  },
  rebuildOn: ['maximumCircumradius', 'groupCount'],
  // The pointer carries the nearest point (it keeps its label) and that group's hull re-forms.
  interaction: {
    kind: 'drag',
    hint: 'carry a point',
    rest: ({inputs, toUnit}) => toUnit([inputs.positions[0], inputs.positions[1]]),
    resolve(pointer, {inputs}) {
      const carried = updatePointCarry(
        inputs.carry,
        inputs.positions,
        [pointer.x, pointer.y],
        0.6,
        1.2
      );
      const group = carried.row >= 0 ? inputs.labels[carried.row] : 0;
      return {
        rerun: carried.changed,
        glyphs: [
          {shape: 'ring', at: carried.at, radius: 0.025},
          {shape: 'dot', at: carried.at}
        ],
        readout: `group ${group} · point ${carried.row} carried`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.groupIds, alpha: 0.55, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.nodeEdges,
      values: outputs.groups,
      radius: 2.4
    }
  ]
});
