// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addRateClusterMapRecipe: event counts and populations per zone become a cluster map of local
// Moran quadrants on the (standardized or smoothed) rate; colors mark high-high, low-high,
// low-low and high-low zones that survive the significance level.

import {
  addRateClusterMapRecipe,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUParameterBuffer
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createBuiltUpZones} from './lib/built-up-zones';
import {getPolygonPort, getQuadFill, uploadQuadZones} from './lib/quads';
import {
  findZoneAt,
  getNeighborhoodGlyphs,
  getNeighborsOf,
  getZoneNeighbors
} from './lib/zone-lookup';
import {createRateDraws, getRateCounts} from './lib/tracts';
import {parameters} from './rate-cluster-map-recipe.parameters';

const QUADRANTS = ['not significant', 'HH', 'LH', 'LL', 'HL'];
/** The palette the recipe is given, in the same order: gray, red, light blue, blue, orange. */
const QUADRANT_COLORS = [
  [0.78, 0.78, 0.78],
  [0.84, 0.1, 0.11],
  [0.67, 0.85, 0.91],
  [0.17, 0.48, 0.71],
  [0.99, 0.68, 0.38]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Significant high and low rate clusters (local Moran quadrants).',
  createInputs: ({random, world}) => {
    const zones = createBuiltUpZones(random, world);
    return {
      zones,
      neighbors: getZoneNeighbors(zones),
      draws: createRateDraws(random, zones.count)
    };
  },
  build(builder, {zones, draws}, values) {
    const events = builder.allocate('events', 'float32', zones.count);
    const populations = builder.allocate('populations', 'float32', zones.count);
    const quadrants = builder.allocate('quadrants', 'uint32', zones.count);
    const mesh = uploadQuadZones(builder, zones);
    const recipe = addRateClusterMapRecipe(builder.graph, {
      events: events.view,
      populations: populations.view,
      polygons: getPolygonPort(mesh),
      criterion: 'rook',
      multipleTesting: 'none',
      neighborCapacity: zones.count * 6,
      analyze: values.analyze === 1 ? 'smoothed' : 'standardized',
      parameters: new GPUParameterBuffer(builder.device, {
        id: 'autocorrelation-parameters',
        format: 'float32',
        length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      }),
      permutation: {
        parameters: new GPUParameterBuffer(builder.device, {
          id: 'permutation-parameters',
          format: 'uint32',
          length: GPU_PERMUTATION_PARAMETER_LENGTH
        }),
        maximumPermutations: 99
      },
      palette: builder.upload(
        'palette',
        'uint32',
        Uint32Array.of(0xffc8c8c8, 0xff1c19d7, 0xffe9d9ab, 0xffb67b2c, 0xff61aefd)
      ).view,
      outputs: {quadrants: quadrants.view}
    });
    return {
      outputs: {quadrants, mesh},
      update: current => {
        const counts = getRateCounts(zones, draws, 4, current.smallShare);
        events.write(Float32Array.from(counts.events));
        populations.write(counts.populations);
        recipe.setSettings({
          significanceLevel: current.significanceLevel,
          seed: 11,
          permutationCount: 99
        });
      }
    };
  },
  rebuildOn: ['analyze'],
  // The pointer picks a zone: its rook neighbors, and the local Moran quadrant it fell in.
  interaction: {
    kind: 'probe',
    hint: 'hover a zone',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.zones.centers[2 * 70], inputs.zones.centers[2 * 70 + 1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const zone = findZoneAt(inputs.zones, pointer.x, pointer.y);
      const around = getNeighborsOf(inputs.neighbors, zone);
      const quadrants = peek(outputs.quadrants);
      return {
        glyphs: getNeighborhoodGlyphs(inputs.zones, zone, around),
        readout: `${quadrants ? (QUADRANTS[quadrants[zone]] ?? 'n.s.') : '...'} · alpha ${values.significanceLevel} · ${around.length} neighbors`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getQuadFill(outputs.mesh, inputs.zones, outputs.quadrants, {categories: QUADRANT_COLORS}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});
