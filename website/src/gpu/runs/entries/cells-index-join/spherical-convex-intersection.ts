// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUSphericalConvexIntersection} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {addGroupedRingSegments, addUnitVectorsToLonLat} from './support/kernels';
import {packUnitVectors} from './support/polygons';
import {
  getConvexShore,
  getLakeShore,
  getRingSegments,
  getSurveyRing,
  SUPERIOR_WINDOW
} from './support/spherical-surveys';
import {parameters} from './spherical-convex-intersection.parameters';

const CAPACITY = 32;
const LONGITUDE_PARAMETER = findParameter(parameters, 'separation');
const LATITUDE_PARAMETER = findParameter(parameters, 'surveyLatitude');

export default defineOperationRun({
  parameters,
  geographicWindow: SUPERIOR_WINDOW,
  caption:
    'A modeled survey footprint intersects the convex envelope of Lake Superior. The mapped shoreline provides context.',
  createInputs: () => {
    const shore = getLakeShore('Lake Superior');
    return {shore, hull: getConvexShore(shore)};
  },
  build(builder, {shore, hull}, values) {
    const sides = values.sides;
    const second = builder.allocate('second-positions', 'float32x3', sides);
    const intersection = builder.allocate('intersection', 'float32x3', CAPACITY);
    const counts = builder.allocate('counts', 'uint32x2', 1);
    builder.graph.add(
      new GPUSphericalConvexIntersection({
        firstPositions: builder.upload('hull', 'float32x3', packUnitVectors([hull])).view,
        secondPositions: second.view,
        firstOffsets: builder.upload('first-offsets', 'uint32', Uint32Array.of(0, hull.length))
          .view,
        secondOffsets: builder.upload('second-offsets', 'uint32', Uint32Array.of(0, sides)).view,
        capacity: CAPACITY,
        output: {
          positions: intersection.view,
          counts: counts.view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    const lonLat = addUnitVectorsToLonLat(builder, 'intersection', intersection);
    const outline = builder.allocate('survey-outline', 'float32x4', sides);
    return {
      outputs: {
        shore: builder.upload('shore', 'float32x4', getRingSegments([shore])),
        hull: builder.upload('hull-outline', 'float32x4', getRingSegments([hull])),
        outline,
        counts,
        ring: addGroupedRingSegments(builder, 'intersection', lonLat, counts, CAPACITY)
      },
      update: current => {
        const ring = getSurveyRing(
          current.separation,
          current.surveyLatitude,
          130,
          sides,
          current.rotation
        );
        second.write(packUnitVectors([ring]));
        outline.write(getRingSegments([ring]));
      }
    };
  },
  rebuildOn: ['sides'],
  // Drag: the survey footprint is centered on the pointer's longitude and latitude.
  interaction: {
    kind: 'drag',
    hint: 'drag the footprint',
    rest: ({toUnit}) => toUnit([LONGITUDE_PARAMETER.default, LATITUDE_PARAMETER.default]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const separation = snapToParameter(LONGITUDE_PARAMETER, pointer.x);
      const surveyLatitude = snapToParameter(LATITUDE_PARAMETER, pointer.y);
      const counts = peek(outputs.counts);
      return {
        values: {separation, surveyLatitude},
        glyphs: [{shape: 'crosshair', at: [separation, surveyLatitude]}],
        readout: `intersection ${counts ? counts[0] : '…'} vertices`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.shore,
      radius: 1.3,
      color: [...theme.muted, 0.9]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.hull,
      radius: 1.2,
      color: [0.45, 0.65, 1, 0.85]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      radius: 1.2,
      color: [1, 0.6, 0.35, 0.85]
    },
    {layer: 'lines', segments: outputs.ring, radius: 3, color: [...theme.ink, 1]}
  ]
});
