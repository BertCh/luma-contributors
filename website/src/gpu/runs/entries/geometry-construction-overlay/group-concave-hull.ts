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
import {addEdgeDrawing} from './shared/edges';
import {parameters} from './group-concave-hull.parameters';

export default defineOperationRun({
  parameters,
  caption: 'One concave hull per label; each boundary has its group color.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 300, 3, 0.08),
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
    return {outputs: {positions, groupIds, ...drawing}};
  },
  rebuildOn: ['maximumCircumradius', 'groupCount'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.nodeEdges,
      values: outputs.groups,
      radius: 2.4
    }
  ]
});
