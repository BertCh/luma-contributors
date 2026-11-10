// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointBuffer generating the ring vertices around points (planar mode, radius in map units). The
// GPU writes the offsets of each ring vertex from its center; the outlines are the rings joined
// vertex to vertex around each point.

import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '@luma-contributors/analysis/geospatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import type {Point} from '../../interaction';
import {addRingSegments} from './_lib/ring-segments';
import {sampleFlashes} from '../../../data/lightning';
import {parameters} from './point-buffer.parameters';

const MAXIMUM_POINT_COUNT = 24;

function getNearestCenter(centers: Float32Array, pointCount: number, x: number, y: number) {
  let best = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < pointCount; index++) {
    const distance = Math.hypot(centers[index * 2] - x, centers[index * 2 + 1] - y);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return {index: best, distance: bestDistance};
}

export default defineOperationRun({
  parameters,
  caption: 'Fixed-radius buffers around measured Midwest lightning flashes.',
  createInputs: ({world}) => ({
    centers: sampleFlashes(world, MAXIMUM_POINT_COUNT).positions
  }),
  build(builder, inputs, values) {
    const pointCount = values.pointCount;
    const centers = builder.upload('centers', 'float32x2', inputs.centers.slice(0, pointCount * 2));
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const offsets = builder.allocate('offsets', 'float32x2', pointCount * values.verticesPerRing);
    builder.graph.add(
      new GPUPointBuffer({
        points: centers.view,
        parameters: parameterBuffer.view,
        offsets: offsets.view,
        verticesPerRing: values.verticesPerRing,
        coordinateSystem: 'planar'
      })
    );
    return {
      outputs: {
        centers,
        rings: addRingSegments(builder, 'rings', centers, offsets, values.verticesPerRing)
      },
      update: current => parameterBuffer.write(getGPUPointBufferParameterValues(current.radius))
    };
  },
  rebuildOn: ['verticesPerRing', 'pointCount'],
  // The pointer sets the radius: its distance to the nearest center, so a ring passes under it.
  interaction: {
    kind: 'reach',
    hint: 'pull a ring out',
    // Rest on the most isolated center, at the default radius from it, so that center stays nearest.
    rest: ({inputs, values, toUnit}) => {
      const radius = findParameter(parameters, 'radius').default;
      let rest: Point = [0.5, 0.5];
      let isolation = -1;
      for (let index = 0; index < values.pointCount; index++) {
        const x = inputs.centers[index * 2];
        const y = inputs.centers[index * 2 + 1];
        for (const angle of [0, Math.PI, Math.PI / 2, -Math.PI / 2]) {
          const candidate: Point = [x + Math.cos(angle) * radius, y + Math.sin(angle) * radius];
          const nearest = getNearestCenter(
            inputs.centers,
            values.pointCount,
            candidate[0],
            candidate[1]
          );
          const [unitX, unitY] = toUnit(candidate);
          if (
            nearest.index === index &&
            unitX > 0.03 &&
            unitX < 0.97 &&
            unitY > 0.03 &&
            unitY < 0.97
          ) {
            const margin = Math.min(unitX, 1 - unitX, unitY, 1 - unitY);
            if (margin > isolation) {
              isolation = margin;
              rest = [unitX, unitY];
            }
          }
        }
      }
      return rest;
    },
    resolve(pointer, {inputs, values}) {
      const nearest = getNearestCenter(inputs.centers, values.pointCount, pointer.x, pointer.y);
      const parameter = findParameter(parameters, 'radius');
      const radius = snapToParameter(
        parameter,
        clamp(nearest.distance, parameter.min, parameter.max)
      );
      const center: Point = [
        inputs.centers[nearest.index * 2],
        inputs.centers[nearest.index * 2 + 1]
      ];
      return {
        values: {radius},
        glyphs: [
          {shape: 'line', from: center, to: [pointer.x, pointer.y], dashed: true},
          {shape: 'dot', at: [pointer.x, pointer.y]}
        ],
        readout: `radius ${radius.toFixed(3)} · ${values.verticesPerRing} vertices per ring`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.centers,
      color: [0.9, 0.9, 1, 0.9],
      radius: 3
    },
    {layer: 'lines', segments: outputs.rings, color: [0.3, 0.75, 1, 1], radius: 1.8}
  ]
});
