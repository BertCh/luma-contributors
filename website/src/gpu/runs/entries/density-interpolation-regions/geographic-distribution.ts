// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicDistribution summarizes each group of points: mean center, median center,
// standard-distance circle and standard deviational ellipse (rings written by the operation).

import {
  getGPUGeographicDistributionParameterValues,
  GPUGeographicDistribution
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './geographic-distribution.parameters';

const GROUP_COUNT = 3;
const POINTS_PER_GROUP = 160;
const RING_VERTEX_COUNT = 64;

function createRingEdges(): Uint32Array {
  const edges = new Uint32Array(GROUP_COUNT * RING_VERTEX_COUNT * 2);
  for (let group = 0; group < GROUP_COUNT; group++) {
    for (let vertex = 0; vertex < RING_VERTEX_COUNT; vertex++) {
      const edge = (group * RING_VERTEX_COUNT + vertex) * 2;
      edges[edge] = group * RING_VERTEX_COUNT + vertex;
      edges[edge + 1] = group * RING_VERTEX_COUNT + ((vertex + 1) % RING_VERTEX_COUNT);
    }
  }
  return edges;
}

export default defineOperationRun({
  parameters,
  caption: 'Per-group center, standard distance circle and directional ellipse.',
  createInputs: ({random, world}) => {
    const positions = new Float32Array(GROUP_COUNT * POINTS_PER_GROUP * 2);
    const groupIds = new Uint32Array(GROUP_COUNT * POINTS_PER_GROUP);
    const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    for (let group = 0; group < GROUP_COUNT; group++) {
      const centerX = ((group + 0.5) / GROUP_COUNT) * world.width;
      const centerY = 0.3 + 0.4 * random();
      const angle = random() * Math.PI;
      for (let point = 0; point < POINTS_PER_GROUP; point++) {
        const along = normal() * 0.1;
        const across = normal() * 0.035;
        const row = group * POINTS_PER_GROUP + point;
        positions[row * 2] = centerX + along * Math.cos(angle) - across * Math.sin(angle);
        positions[row * 2 + 1] = centerY + along * Math.sin(angle) + across * Math.cos(angle);
        groupIds[row] = group;
      }
    }
    const centers = Float32Array.from({length: GROUP_COUNT * 2}, (_, index) => {
      let sum = 0;
      for (let point = 0; point < POINTS_PER_GROUP; point++) {
        sum += positions[(Math.floor(index / 2) * POINTS_PER_GROUP + point) * 2 + (index % 2)];
      }
      return sum / POINTS_PER_GROUP;
    });
    return {positions, groupIds, centers};
  },
  build(builder, {positions, groupIds, centers}) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const points = builder.upload('points', 'float32x2', positions);
    const output = {
      meanCenters: builder.allocate('mean-centers', 'float32x2', GROUP_COUNT),
      medianCenters: builder.allocate('median-centers', 'float32x2', GROUP_COUNT),
      ellipseVertices: builder.allocate(
        'ellipse-vertices',
        'float32x2',
        GROUP_COUNT * RING_VERTEX_COUNT
      ),
      circleVertices: builder.allocate(
        'circle-vertices',
        'float32x2',
        GROUP_COUNT * RING_VERTEX_COUNT
      )
    };
    builder.graph.add(
      new GPUGeographicDistribution({
        positions: points.view,
        groupIds: builder.upload('group-ids', 'uint32', groupIds).view,
        groupCount: GROUP_COUNT,
        parameters: parameterBuffer.view,
        polygonVertexCount: RING_VERTEX_COUNT,
        output: {
          meanCenters: output.meanCenters.view,
          medianCenters: output.medianCenters.view,
          ellipseVertices: output.ellipseVertices.view,
          circleVertices: output.circleVertices.view
        }
      })
    );
    const ringGroups = Uint32Array.from({length: GROUP_COUNT * RING_VERTEX_COUNT}, (_, vertex) =>
      Math.floor(vertex / RING_VERTEX_COUNT)
    );
    return {
      outputs: {
        ...output,
        points,
        groups: builder.upload('groups', 'uint32', groupIds),
        ringGroups: builder.upload('ring-groups', 'uint32', ringGroups),
        ringEdges: builder.upload('ring-edges', 'uint32x2', createRingEdges())
      },
      update: values => {
        points.write(
          Float32Array.from({length: positions.length}, (_, index) => {
            const center = centers[Math.floor(index / 2 / POINTS_PER_GROUP) * 2 + (index % 2)];
            return center + (positions[index] - center) * values.spread;
          })
        );
        parameterBuffer.write(
          getGPUGeographicDistributionParameterValues({
            standardDeviations: values.standardDeviations,
            ellipseConvention: values.ellipseConvention === 0 ? 'arcgis' : 'standard'
          })
        );
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, values: outputs.groups, radius: 2.2},
    {
      layer: 'segments',
      nodes: outputs.circleVertices,
      edges: outputs.ringEdges,
      values: outputs.ringGroups,
      alpha: 0.5
    },
    {
      layer: 'segments',
      nodes: outputs.ellipseVertices,
      edges: outputs.ringEdges,
      values: outputs.ringGroups
    },
    {layer: 'dots', positions: outputs.meanCenters, radius: 6, color: [1, 1, 1, 1]},
    {layer: 'dots', positions: outputs.medianCenters, radius: 4, color: [1, 0.85, 0.3, 1]}
  ]
});
