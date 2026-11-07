// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGroupGeometry summarizes every label group: bounding box, mean center, medoid and standard
// distance. Points are colored by label; boxes (orange), the standard-distance circle (blue), mean
// centers and medoids (large dots) are drawn from the buffers the operation wrote.

import {
  getGPUGeographicDistributionParameterValues,
  GPUGroupGeometry
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {addKernelPass} from './shared/kernel';
import {HIGHLIGHT_COLOR, OUTPUT_COLOR} from './shared/lines';
import {parameters} from './group-geometry.parameters';

const CIRCLE_SEGMENTS = 40;
const SEGMENTS_PER_GROUP = 4 + CIRCLE_SEGMENTS;

export default defineOperationRun({
  parameters,
  caption: 'Per-group bounds, mean center, medoid and standard distance of labeled points.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 360, 4, 0.08),
  build(builder, points, values) {
    const groupCount = values.groupCount;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const labels = builder.upload('labels', 'uint32', points.labels);
    const parameterBuffer = builder.allocate('distribution-parameters', 'float32', 8);
    const counts = builder.allocate('counts', 'uint32', groupCount);
    const bounds = builder.allocate('bounds', 'float32x4', groupCount);
    const meanCenters = builder.allocate('mean-centers', 'float32x2', groupCount);
    const medoidIndices = builder.allocate('medoid-indices', 'uint32', groupCount);
    const standardDistances = builder.allocate('standard-distances', 'float32', groupCount);
    builder.graph.add(
      new GPUGroupGeometry({
        positions: positions.view,
        labels: labels.view,
        groupCount,
        noiseLabel: 4,
        parameters: parameterBuffer.view,
        output: {
          counts: counts.view,
          bounds: bounds.view,
          meanCenters: meanCenters.view,
          medoidIndices: medoidIndices.view,
          standardDistances: standardDistances.view
        }
      })
    );
    // Drawing: four box edges and a circle of the standard distance per group, plus medoid points.
    const boxes = builder.allocate('boxes', 'float32x4', groupCount * 4);
    const circles = builder.allocate('circles', 'float32x4', groupCount * CIRCLE_SEGMENTS);
    const medoids = builder.allocate('medoids', 'float32x2', groupCount);
    addKernelPass(builder, {
      id: 'group-shape-pass',
      invocationCount: groupCount * SEGMENTS_PER_GROUP,
      bindings: [
        {name: 'bounds', buffer: bounds, access: 'read'},
        {name: 'meanCenters', buffer: meanCenters, access: 'read'},
        {name: 'standardDistances', buffer: standardDistances, access: 'read'},
        {name: 'boxes', buffer: boxes, access: 'read_write'},
        {name: 'circles', buffer: circles, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let group = index / ${SEGMENTS_PER_GROUP}u;
  let slot = index % ${SEGMENTS_PER_GROUP}u;
  let box = bounds[boundsOffset + group];
  if (slot < 4u) {
    var corners = array<vec2f, 4>(box.xy, vec2f(box.z, box.y), box.zw, vec2f(box.x, box.w));
    boxes[boxesOffset + group * 4u + slot] = vec4f(corners[slot], corners[(slot + 1u) % 4u]);
  } else {
    let step = slot - 4u;
    let center = meanCenters[meanCentersOffset + group];
    let radius = standardDistances[standardDistancesOffset + group];
    let first = 6.2831853 * f32(step) / ${CIRCLE_SEGMENTS}.0;
    let second = 6.2831853 * f32(step + 1u) / ${CIRCLE_SEGMENTS}.0;
    circles[circlesOffset + group * ${CIRCLE_SEGMENTS}u + step] = vec4f(
      center + radius * vec2f(cos(first), sin(first)),
      center + radius * vec2f(cos(second), sin(second)));
  }`
    });
    addKernelPass(builder, {
      id: 'group-medoid-pass',
      invocationCount: groupCount,
      bindings: [
        {name: 'positions', buffer: positions, access: 'read'},
        {name: 'medoidIndices', buffer: medoidIndices, access: 'read'},
        {name: 'medoids', buffer: medoids, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let row = medoidIndices[medoidIndicesOffset + index];
  medoids[medoidsOffset + index] = select(vec2f(-9.0), positions[positionsOffset + row], row != 0xffffffffu);`
    });
    return {
      outputs: {positions, labels, boxes, circles, meanCenters, medoids},
      update: current =>
        parameterBuffer.write(
          getGPUGeographicDistributionParameterValues({
            standardDeviations: current.standardDeviations
          })
        )
    };
  },
  rebuildOn: ['groupCount'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 3},
    {layer: 'lines', segments: outputs.boxes, color: HIGHLIGHT_COLOR, radius: 1.8},
    {layer: 'lines', segments: outputs.circles, color: OUTPUT_COLOR, radius: 2},
    {layer: 'dots', positions: outputs.meanCenters, color: OUTPUT_COLOR, radius: 7},
    {layer: 'dots', positions: outputs.medoids, color: HIGHLIGHT_COLOR, radius: 5}
  ]
});
