// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUSphericalConvexContains} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {HONSHU_WINDOW, getHonshuEvents} from '../../../data/point-observations';
import coastline from '../../../data/japan-coast.json';
import {defineOperationRun} from '../../define-run';
import {formatLongitudeLatitude} from './support/pointer';
import {packUnitVectors} from './support/polygons';
import {getSurveyRing, getRingSegments} from './support/spherical-surveys';
import {parameters} from './spherical-convex-contains.parameters';

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption:
    'Measured offshore Honshu epicenters inside a modeled geodesic search area that follows the pointer.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 700),
    center: [142.15, 37.45] as [number, number]
  }),
  build(builder, inputs, values) {
    const events = inputs;
    const sides = values.sides;
    const positions = builder.allocate('positions', 'float32x3', sides);
    const outline = builder.allocate('outline', 'float32x4', sides);
    const contains = builder.allocate('contains', 'uint32', events.count);
    const queries = Array.from({length: events.count}, (_, index) => [
      events.positions[index * 2],
      events.positions[index * 2 + 1]
    ]);
    builder.graph.add(
      new GPUSphericalConvexContains({
        positions: positions.view,
        offsets: builder.upload('offsets', 'uint32', Uint32Array.of(0, sides)).view,
        queries: builder.upload('queries-xyz', 'float32x3', packUnitVectors([queries])).view,
        polygonIndices: builder.upload('polygon-indices', 'uint32', new Uint32Array(events.count))
          .view,
        // Sub-degree f32 arcs need a tolerance larger than edge-normal roundoff.
        tolerance: 3e-5,
        complement: values.complement === 1,
        output: {
          contains: contains.view,
          status: builder.allocate('status', 'uint32', events.count).view
        }
      })
    );
    return {
      outputs: {
        contains,
        outline,
        queries: builder.upload('queries', 'float32x2', events.positions),
        coast: builder.upload('coast', 'float32x4', Float32Array.from(coastline.flat()))
      },
      update: current => {
        const ring = getSurveyRing(inputs.center[0], inputs.center[1], current.size, sides);
        positions.write(packUnitVectors([ring]));
        outline.write(getRingSegments([ring]));
      }
    };
  },
  rebuildOn: ['complement', 'sides'],
  // Place: the geodesic search polygon is centered on the pointer's longitude and latitude.
  interaction: {
    kind: 'place',
    hint: 'move the search area',
    rest: ({toUnit}) => toUnit([142.15, 37.45]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.center = [pointer.x, pointer.y];
      const contains = peek(outputs.contains);
      let inside = 0;
      if (contains) for (const flag of contains) inside += flag > 0 ? 1 : 0;
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: inputs.center, size: 6}],
        readout: `center ${formatLongitudeLatitude(pointer.x, pointer.y)} · ${values.size} km · ${contains ? inside : '…'} inside`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.coast,
      radius: 1.2,
      color: [...theme.muted, 0.9]
    },
    {layer: 'dots', positions: outputs.queries, values: outputs.contains, radius: 4.6},
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      radius: 1.8,
      color: [...theme.ink, 0.9]
    }
  ]
});
