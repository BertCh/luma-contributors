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
import {findParameter, snapToParameter} from '../../interaction';
import {addKernelPass} from './shared/kernel';
import {HIGHLIGHT_COLOR, OUTPUT_COLOR} from './shared/lines';
import {parameters} from './group-geometry.parameters';

const CIRCLE_SEGMENTS = 40;
// The operation's default 'arcgis' convention scales the ellipse axes by sqrt(2).
const ELLIPSE_SCALE = Math.SQRT2;
const SEGMENTS_PER_GROUP = 4 + CIRCLE_SEGMENTS;
const GROUP_LABELS = 4;

type GroupSummary = {center: [number, number]; angle: number; long: number; short: number};

/** The ellipse (axes scaled by `size` standard deviations) point at parameter `turn` around it. */
function getEllipsePoint(group: GroupSummary, turn: number, size: number): [number, number] {
  const longAxis = group.long * size * ELLIPSE_SCALE * Math.cos(turn);
  const shortAxis = group.short * size * ELLIPSE_SCALE * Math.sin(turn);
  return [
    group.center[0] + longAxis * Math.cos(group.angle) - shortAxis * Math.sin(group.angle),
    group.center[1] + longAxis * Math.sin(group.angle) + shortAxis * Math.cos(group.angle)
  ];
}

/** The number of standard deviations at which the group's ellipse passes through (x, y). */
function getEllipseSize(group: GroupSummary, x: number, y: number): number {
  const deltaX = x - group.center[0];
  const deltaY = y - group.center[1];
  const alongLong = deltaX * Math.cos(group.angle) + deltaY * Math.sin(group.angle);
  const alongShort = -deltaX * Math.sin(group.angle) + deltaY * Math.cos(group.angle);
  return Math.hypot(alongLong / group.long, alongShort / group.short) / ELLIPSE_SCALE;
}

function getNearestGroup(
  groups: {center: [number, number]}[],
  groupCount: number,
  x: number,
  y: number
) {
  let nearest = 0;
  let nearestDistance = Infinity;
  for (let group = 0; group < groupCount; group++) {
    const distance = Math.hypot(x - groups[group].center[0], y - groups[group].center[1]);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = group;
    }
  }
  return {group: nearest, distance: nearestDistance};
}

/** Mean center and standard distance of each label group, as the operation defines them. */
function summarizeGroups(positions: Float32Array, labels: Uint32Array) {
  return Array.from({length: GROUP_LABELS}, (_, group) => {
    let count = 0;
    let sumX = 0;
    let sumY = 0;
    labels.forEach((label, row) => {
      if (label !== group) return;
      count++;
      sumX += positions[row * 2];
      sumY += positions[row * 2 + 1];
    });
    const center: [number, number] = [sumX / count, sumY / count];
    let varianceX = 0;
    let varianceY = 0;
    let covariance = 0;
    labels.forEach((label, row) => {
      if (label !== group) return;
      const deltaX = positions[row * 2] - center[0];
      const deltaY = positions[row * 2 + 1] - center[1];
      varianceX += (deltaX * deltaX) / count;
      varianceY += (deltaY * deltaY) / count;
      covariance += (deltaX * deltaY) / count;
    });
    const middle = (varianceX + varianceY) / 2;
    const spread = Math.hypot((varianceX - varianceY) / 2, covariance);
    return {
      center,
      /** Direction of the long axis and the principal standard deviations. */
      angle: 0.5 * Math.atan2(2 * covariance, varianceX - varianceY),
      long: Math.sqrt(middle + spread),
      short: Math.sqrt(Math.max(middle - spread, 1e-9))
    };
  });
}

export default defineOperationRun({
  parameters,
  caption: 'Per-group bounds, mean center, medoid and standard distance of labeled points.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 360, 4, 0.08);
    return {...points, groups: summarizeGroups(points.positions, points.labels), world};
  },
  build(builder, points, values) {
    const groupCount = values.groupCount;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const labels = builder.upload('labels', 'uint32', points.labels);
    const parameterBuffer = builder.allocate('distribution-parameters', 'float32', 8);
    const counts = builder.allocate('counts', 'uint32', groupCount);
    const bounds = builder.allocate('bounds', 'float32x4', groupCount);
    const meanCenters = builder.allocate('mean-centers', 'float32x2', groupCount);
    const medoidIndices = builder.allocate('medoid-indices', 'uint32', groupCount);
    const ellipses = builder.allocate('ellipses', 'float32', groupCount * 3);
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
          ellipses: ellipses.view
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
        {name: 'ellipses', buffer: ellipses, access: 'read'},
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
    let angle = ellipses[ellipsesOffset + group * 3u];
    let axes = vec2f(ellipses[ellipsesOffset + group * 3u + 1u], ellipses[ellipsesOffset + group * 3u + 2u]);
    let rotation = mat2x2f(cos(angle), sin(angle), -sin(angle), cos(angle));
    let first = 6.2831853 * f32(step) / ${CIRCLE_SEGMENTS}.0;
    let second = 6.2831853 * f32(step + 1u) / ${CIRCLE_SEGMENTS}.0;
    circles[circlesOffset + group * ${CIRCLE_SEGMENTS}u + step] = vec4f(
      center + rotation * (axes * vec2f(cos(first), sin(first))),
      center + rotation * (axes * vec2f(cos(second), sin(second))));
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
  // The pointer sets the size of the nearest group's standard deviational ellipse: the ellipse
  // (in its own axes) passes through the cursor.
  interaction: {
    kind: 'reach',
    hint: 'move away from a center',
    rest: ({inputs, toUnit}) => {
      const target = findParameter(parameters, 'standardDeviations').default;
      // A spot where the nearest center is the group whose circle passes through it.
      for (const group of inputs.groups) {
        for (let step = 0; step < 24; step++) {
          const angle = (step / 24) * Math.PI * 2;
          const point = getEllipsePoint(group, angle, target);
          const [x, y] = point;
          const nearest = getNearestGroup(inputs.groups, GROUP_LABELS, x, y);
          if (
            inputs.groups[nearest.group] === group &&
            x > 0.03 &&
            x < inputs.world.width - 0.03 &&
            y > 0.03 &&
            y < 0.97
          ) {
            return toUnit([x, y]);
          }
        }
      }
      return toUnit(inputs.groups[0].center);
    },
    resolve(pointer, {inputs, values}) {
      const {group: nearest} = getNearestGroup(
        inputs.groups,
        Math.min(values.groupCount, GROUP_LABELS),
        pointer.x,
        pointer.y
      );
      const group = inputs.groups[nearest];
      const {center} = group;
      const standardDeviations = snapToParameter(
        findParameter(parameters, 'standardDeviations'),
        getEllipseSize(group, pointer.x, pointer.y)
      );
      return {
        values: {standardDeviations},
        glyphs: [
          {shape: 'line', from: center, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: center}
        ],
        readout: `group ${nearest} · standardDeviations ${standardDeviations.toFixed(1)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 3},
    {layer: 'lines', segments: outputs.boxes, color: HIGHLIGHT_COLOR, radius: 1, alpha: 0.7},
    {layer: 'lines', segments: outputs.circles, color: OUTPUT_COLOR, radius: 2},
    {layer: 'dots', positions: outputs.meanCenters, color: OUTPUT_COLOR, radius: 7},
    {layer: 'dots', positions: outputs.medoids, color: HIGHLIGHT_COLOR, radius: 5}
  ]
});
