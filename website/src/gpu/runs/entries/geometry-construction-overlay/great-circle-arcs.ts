// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGreatCircleArcs turns origin and destination longitude/latitude pairs into arcs along the
// great circle. On this flat (equirectangular) picture the arcs bow toward the poles, while the
// muted straight lines are the flat-map shortcut.

import {createCoastlineSegments} from '../../../data/coastline';
import {cityCoordinates} from '../../../data/hydrography';
import {
  getGPUGreatCircleArcsParameterValues,
  GPUGreatCircleArcs
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  allocateLineOutput,
  type DrawTransform,
  HIGHLIGHT_COLOR,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './great-circle-arcs.parameters';

// Every arc flies to one destination, which the pointer carries.
const ORIGINS = [
  'New York',
  'Los Angeles',
  'São Paulo',
  'Cape Town',
  'Beijing',
  'Sydney',
  'Tokyo'
] as const;
const HUB = 'Cairo';
const PAIR_COUNT = ORIGINS.length;
const EARTH_RADIUS_KM = 6371;

function getDistanceKm(from: ArrayLike<number>, to: ArrayLike<number>): number {
  const radians = Math.PI / 180;
  const a =
    Math.sin(((to[1] - from[1]) * radians) / 2) ** 2 +
    Math.cos(from[1] * radians) *
      Math.cos(to[1] * radians) *
      Math.sin(((to[0] - from[0]) * radians) / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(a));
}

export default defineOperationRun({
  parameters,
  caption: 'Great-circle arcs between city pairs; muted lines are the straight flat-map paths.',
  createInputs: ({world}) => {
    const sources = new Float32Array(PAIR_COUNT * 2);
    const targets = new Float32Array(PAIR_COUNT * 2);
    const flat = new Float32Array(PAIR_COUNT * 4);
    const transform: DrawTransform = {
      scale: [world.width / 360, 1 / 150],
      offset: [world.width / 2, 0.5]
    };
    for (let pair = 0; pair < PAIR_COUNT; pair++) {
      sources.set(cityCoordinates[ORIGINS[pair]], pair * 2);
      targets.set(cityCoordinates[HUB], pair * 2);
    }
    const writeFlat = () => {
      for (let pair = 0; pair < PAIR_COUNT; pair++) {
        flat.set(
          [
            sources[pair * 2] * transform.scale[0] + transform.offset[0],
            sources[pair * 2 + 1] * transform.scale[1] + transform.offset[1],
            targets[pair * 2] * transform.scale[0] + transform.offset[0],
            targets[pair * 2 + 1] * transform.scale[1] + transform.offset[1]
          ],
          pair * 4
        );
      }
    };
    writeFlat();
    return {sources, targets, flat, transform, writeFlat, hub: cityCoordinates[HUB]};
  },
  build(builder, inputs) {
    const sources = builder.upload('sources', 'float32x2', inputs.sources);
    const targets = builder.upload('targets', 'float32x2', inputs.targets);
    const result = allocateLineOutput(builder, 'arcs', 4000, PAIR_COUNT);
    const parameterBuffer = builder.allocate('arc-parameters', 'float32', 4);
    builder.graph.add(
      new GPUGreatCircleArcs({
        sources: sources.view,
        targets: targets.view,
        parameters: parameterBuffer.view,
        output: result.output
      })
    );
    const drawing = addPackedSegments(builder, 'arcs', result.positions, result.lineOffsets, {
      transform: inputs.transform
    });
    const coastline = builder.upload(
      'coastline',
      'float32x4',
      createCoastlineSegments(inputs.transform.scale, inputs.transform.offset)
    );
    const flat = builder.upload('flat', 'float32x4', inputs.flat);
    return {
      outputs: {flat, coastline, ...drawing},
      update: current => {
        targets.write(inputs.targets);
        flat.write(inputs.flat);
        parameterBuffer.write(
          getGPUGreatCircleArcsParameterValues({
            maximumSegmentLength: current.maximumSegmentLength * 1000,
            minimumSegments: current.minimumSegments
          })
        );
      }
    };
  },
  // The pointer is the destination of every arc: it is unprojected to longitude and latitude and
  // written into the targets, so all seven great circles are recomputed for the cursor.
  interaction: {
    kind: 'place',
    hint: 'move the destination',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.hub[0] * inputs.transform.scale[0] + inputs.transform.offset[0],
        inputs.hub[1] * inputs.transform.scale[1] + inputs.transform.offset[1]
      ]),
    resolve(pointer, {inputs}) {
      const longitude = (pointer.x - inputs.transform.offset[0]) / inputs.transform.scale[0];
      const latitude = (pointer.y - inputs.transform.offset[1]) / inputs.transform.scale[1];
      let longest = 0;
      for (let pair = 0; pair < PAIR_COUNT; pair++) {
        inputs.targets[pair * 2] = longitude;
        inputs.targets[pair * 2 + 1] = latitude;
        longest = Math.max(
          longest,
          getDistanceKm(
            [inputs.sources[pair * 2], inputs.sources[pair * 2 + 1]],
            [longitude, latitude]
          )
        );
      }
      inputs.writeFlat();
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `to ${Math.abs(longitude).toFixed(0)}°${longitude < 0 ? 'W' : 'E'} ${Math.abs(latitude).toFixed(0)}°${latitude < 0 ? 'S' : 'N'} · longest ${Math.round(longest).toLocaleString('en-US')} km`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.coastline, radius: 0.8, alpha: 0.45},
    {layer: 'lines', context: true, segments: outputs.flat},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 2.9},
    {layer: 'dots', positions: outputs.points, radius: 3.8, color: HIGHLIGHT_COLOR}
  ]
});
