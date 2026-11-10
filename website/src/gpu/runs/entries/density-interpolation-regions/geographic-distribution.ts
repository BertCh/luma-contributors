// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicDistribution summarizes each group of points: mean center, median center,
// standard-distance circle and standard deviational ellipse (rings written by the operation).

import {
  getGPUGeographicDistributionParameterValues,
  GPUGeographicDistribution
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import events from '../../../data/kilauea-events.json';
import {projectObservedPositions} from '../../../data/observed-positions';
import {findParameter, snapToParameter} from '../../interaction';
import {parameters} from './geographic-distribution.parameters';

const GROUP_COUNT = 3;
const RING_VERTEX_COUNT = 64;
// Pointer distance, in world units, that stands for the largest standardDeviations (3).
const REACH = 0.9;

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
  caption: 'Kīlauea epicenter dispersion in successive 40-day windows.',
  createInputs: ({world}) => {
    // Drop the 8% most distant epicenters on each axis, so the frame fits the dense windows.
    const trimmed = trimOutliers(events);
    const positions = projectObservedPositions(trimmed, world);
    const groupIds = Uint32Array.from(trimmed, event =>
      Math.min(GROUP_COUNT - 1, Math.floor(event[2] / 40))
    );
    const counts = new Uint32Array(GROUP_COUNT);
    const centers = new Float32Array(GROUP_COUNT * 2);
    groupIds.forEach((group, row) => {
      counts[group]++;
      centers[group * 2] += positions[row * 2];
      centers[group * 2 + 1] += positions[row * 2 + 1];
    });
    for (let index = 0; index < centers.length; index++)
      centers[index] /= counts[Math.floor(index / 2)];
    // Standard distance of every group: the radius of the circle at standardDeviations = 1.
    const distances = new Float32Array(GROUP_COUNT);
    groupIds.forEach((group, row) => {
      distances[group] +=
        (positions[row * 2] - centers[group * 2]) ** 2 +
        (positions[row * 2 + 1] - centers[group * 2 + 1]) ** 2;
    });
    distances.forEach((sum, group) => (distances[group] = Math.sqrt(sum / counts[group])));
    // The largest group anchors the gesture.
    const anchor = counts.indexOf(Math.max(...counts));
    return {positions, groupIds, centers, distances, anchor};
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
            const center = centers[groupIds[Math.floor(index / 2)] * 2 + (index % 2)];
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
  // The distance from the anchor group's mean center to the pointer sets standardDeviations: the
  // line reaches from the center to the ring that many standard distances out.
  interaction: {
    kind: 'reach',
    hint: 'reach out from the center',
    rest: ({inputs, toUnit}) => {
      const x = inputs.centers[inputs.anchor * 2];
      return toUnit([x + REACH / 3, inputs.centers[inputs.anchor * 2 + 1]]);
    },
    resolve(pointer, {inputs, values}) {
      const {anchor, centers, distances} = inputs;
      const center: [number, number] = [centers[anchor * 2], centers[anchor * 2 + 1]];
      const standardDeviations = snapToParameter(
        findParameter(parameters, 'standardDeviations'),
        (Math.hypot(pointer.x - center[0], pointer.y - center[1]) / REACH) * 3
      );
      const angle = Math.atan2(pointer.y - center[1], pointer.x - center[0]);
      const reach = standardDeviations * distances[anchor];
      return {
        values: {standardDeviations},
        glyphs: [
          {
            shape: 'line',
            from: center,
            to: [center[0] + Math.cos(angle) * reach, center[1] + Math.sin(angle) * reach]
          },
          {shape: 'dot', at: center}
        ],
        readout: `group ${anchor} · standardDeviations ${standardDeviations.toFixed(1)} · ${values.ellipseConvention === 0 ? 'arcgis' : 'standard'}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, values: outputs.groups, radius: 3},
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

function trimOutliers(events: number[][]): number[][] {
  const bounds = [0, 1].map(axis => {
    const sorted = events.map(event => event[axis]).sort((first, second) => first - second);
    return [sorted[Math.floor(sorted.length * 0.08)], sorted[Math.ceil(sorted.length * 0.92) - 1]];
  });
  return events.filter(event =>
    [0, 1].every(axis => event[axis] >= bounds[axis][0] && event[axis] <= bounds[axis][1])
  );
}
