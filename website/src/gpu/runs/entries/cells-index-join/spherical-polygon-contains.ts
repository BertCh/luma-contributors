// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUSphericalPolygonContains} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {sampleFlashes} from '../../../data/lightning';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {packUnitVectors} from './support/polygons';
import {
  getLakeShore,
  getRingSegments,
  getSurveyRing,
  MICHIGAN_WINDOW
} from './support/spherical-surveys';
import {formatLongitudeLatitude} from './support/pointer';
import {parameters} from './spherical-polygon-contains.parameters';

const EXCLUSION_SIDES = 12;
const LONGITUDE_PARAMETER = findParameter(parameters, 'exclusionLongitude');
const LATITUDE_PARAMETER = findParameter(parameters, 'size');

export default defineOperationRun({
  parameters,
  geographicWindow: MICHIGAN_WINDOW,
  caption:
    'GOES lightning detections inside Lake Michigan, excluding an adjustable survey zone. The mapped exterior omits islands.',
  createInputs: () => {
    // The GLM fixture starts at 99° W, 36° N and covers a 16° by 10° window.
    const flashes = sampleFlashes({width: 16, height: 10}, 4096);
    const [west, south, east, north] = MICHIGAN_WINDOW;
    const queries: number[][] = [];
    for (let row = 0; row < flashes.count; row++) {
      const longitude = flashes.positions[row * 2] - 99;
      const latitude = flashes.positions[row * 2 + 1] + 36;
      if (longitude >= west && longitude <= east && latitude >= south && latitude <= north) {
        queries.push([longitude, latitude]);
      }
    }
    return {queries, shore: getLakeShore('Lake Michigan')};
  },
  build(builder, {queries, shore}, values) {
    const vertexCount = shore.length + EXCLUSION_SIDES;
    const positions = builder.allocate('positions', 'float32x3', vertexCount);
    const exclusionOutline = builder.allocate('exclusion-outline', 'float32x4', EXCLUSION_SIDES);
    const shoreOutline = builder.upload('shore-outline', 'float32x4', getRingSegments([shore]));
    const contains = builder.allocate('contains', 'uint32', queries.length);
    builder.graph.add(
      new GPUSphericalPolygonContains({
        positions: positions.view,
        ringOffsets: builder.upload(
          'ring-offsets',
          'uint32',
          Uint32Array.of(0, shore.length, vertexCount)
        ).view,
        polygonOffsets: builder.upload('polygon-offsets', 'uint32', Uint32Array.of(0, 2)).view,
        queries: builder.upload('queries-xyz', 'float32x3', packUnitVectors([queries])).view,
        polygonIndices: builder.upload('polygon-indices', 'uint32', new Uint32Array(queries.length))
          .view,
        complement: values.complement === 1,
        output: {
          contains: contains.view,
          status: builder.allocate('status', 'uint32', queries.length).view
        }
      })
    );
    return {
      outputs: {
        contains,
        exclusionOutline,
        shoreOutline,
        queries: builder.upload('queries', 'float32x2', Float32Array.from(queries.flat()))
      },
      update: current => {
        const exclusion = getSurveyRing(
          current.exclusionLongitude,
          current.size,
          current.holeSize,
          EXCLUSION_SIDES
        );
        const rings = [shore, exclusion];
        positions.write(packUnitVectors(rings));
        exclusionOutline.write(getRingSegments([exclusion]));
      }
    };
  },
  rebuildOn: ['complement'],
  // Place: the exclusion hole is centered on the pointer's longitude and latitude.
  interaction: {
    kind: 'place',
    hint: 'place the exclusion',
    rest: ({toUnit}) => toUnit([LONGITUDE_PARAMETER.default, LATITUDE_PARAMETER.default]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const exclusionLongitude = snapToParameter(LONGITUDE_PARAMETER, pointer.x);
      const size = snapToParameter(LATITUDE_PARAMETER, pointer.y);
      const contains = peek(outputs.contains);
      let inside = 0;
      if (contains) for (const flag of contains) inside += flag > 0 ? 1 : 0;
      return {
        values: {exclusionLongitude, size},
        glyphs: [{shape: 'crosshair', at: [exclusionLongitude, size], size: 6}],
        readout: `hole ${formatLongitudeLatitude(exclusionLongitude, size)} · ${values.holeSize} km · ${contains ? inside : '…'} inside`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.shoreOutline,
      radius: 1.7,
      color: [...theme.ink, 0.85]
    },
    {layer: 'dots', positions: outputs.queries, values: outputs.contains, radius: 5},
    {
      layer: 'lines',
      context: true,
      segments: outputs.exclusionOutline,
      radius: 4.2,
      color: [0.98, 0.53, 0.22, 1]
    }
  ]
});
