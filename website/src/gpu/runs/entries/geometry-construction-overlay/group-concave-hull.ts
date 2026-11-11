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
import {getCentroid, getReachLevel, getReachRest} from './shared/reach';
import {addEdgeDrawing} from './shared/edges';
import {createLegend} from '../../text';
import {parameters} from './group-concave-hull.parameters';

// The circumradius limits the pointer's reach picks among.
const LEVELS = [0.04, 0.1, 0.16, 0.26, 0.4] as const;
const REST_LEVEL = 1;

export default defineOperationRun({
  parameters,
  caption: 'One concave hull per label; each boundary has its group color.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 300, 3, 0.08);
    return {...points, anchor: getCentroid(points.positions)};
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
      outputs: {positions, groupIds, ...drawing}
    };
  },
  rebuildOn: ['maximumCircumradius', 'groupCount'],
  buildCache: 5,
  // The distance from the centroid of all groups to the pointer picks the circumradius limit among
  // five values (applied once it rests); the dashed ball at the cursor is that limit.
  interaction: {
    kind: 'reach',
    hint: 'reach out to loosen the hulls',
    rest: ({inputs, toUnit}) => toUnit(getReachRest(inputs.anchor, REST_LEVEL, LEVELS.length)),
    resolve(pointer, {inputs, values}) {
      const level = getReachLevel(inputs.anchor, [pointer.x, pointer.y], LEVELS.length);
      return {
        settle: {maximumCircumradius: LEVELS[level]},
        glyphs: [
          {shape: 'dot', at: inputs.anchor},
          {shape: 'line', from: inputs.anchor, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'ring', at: [pointer.x, pointer.y], radius: LEVELS[level], dashed: true}
        ],
        readout: `maximumCircumradius ${LEVELS[level].toFixed(2)} · ${values.groupCount} groups`
      };
    }
  },
  draw: ({outputs, values}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.groupIds, alpha: 0.55, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.nodeEdges,
      values: outputs.groups,
      radius: 2.4
    },
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        title: `circumradius limit ${values.maximumCircumradius.toFixed(2)}`,
        entries: [
          ...Array.from({length: values.groupCount}, (_, group) => ({
            label: `group ${group + 1} hull`,
            swatch: group
          })),
          // The scattered 8% of the points belong to no group and take the fourth color.
          {label: 'noise point', swatch: 3}
        ]
      })
    }
  ]
});
