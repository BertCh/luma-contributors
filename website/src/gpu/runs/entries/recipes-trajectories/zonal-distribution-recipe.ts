// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addZonalDistributionRecipe: points joined to zones, then the distribution of their values per
// zone. Zones are filled by the chosen quantile (0.5 is the median); points are the muted input.

import {addZonalDistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {createClusteredPoints} from '../../inputs';
import {createQuadZones, getPolygonPort, getQuadFill, uploadQuadZones} from './lib/quads';
import {findZoneAt, getZoneRing} from './lib/zone-lookup';
import {parameters} from './zonal-distribution-recipe.parameters';

const POINT_COUNT = 2500;

export default defineOperationRun({
  parameters,
  caption: 'Quantile of the point values inside each zone.',
  createInputs: ({random, world}) => ({
    world,
    zones: createQuadZones(random, world, 12, 7),
    points: createClusteredPoints(random, world, POINT_COUNT, 4, 0.2),
    draws: Float32Array.from({length: POINT_COUNT}, () => random() + random() - 1)
  }),
  build(builder, {zones, points, world, draws}) {
    const mesh = uploadQuadZones(builder, zones);
    const pointPositions = builder.upload('points', 'float32x2', points.positions);
    const pointValues = builder.allocate('point-values', 'float32', POINT_COUNT);
    const probabilities = builder.allocate('probabilities', 'float32', 1);
    const quantiles = builder.allocate('quantiles', 'float32', zones.count);
    addZonalDistributionRecipe(builder.graph, {
      source: {
        kind: 'points',
        join: {
          points: {kind: 'points', positions: pointPositions.view},
          polygons: getPolygonPort(mesh),
          candidateCapacity: 1 << 16
        }
      },
      values: pointValues.view,
      probabilities: probabilities.view,
      outputs: {quantiles: quantiles.view}
    });
    return {
      outputs: {quantiles, mesh, pointPositions},
      update: current => {
        probabilities.write(Float32Array.of(current.probability));
        pointValues.write(
          Float32Array.from(
            points.values,
            (value, index) =>
              10 * (value + current.trend * (points.positions[index * 2] / world.width) * 0.8) +
              current.spread * draws[index] * 4
          )
        );
      }
    };
  },
  // x is the probability of the quantile; the zone under the pointer reports its own value.
  interaction: {
    kind: 'level',
    hint: 'slide the quantile',
    rest: ({toUnit, inputs}) => [
      getFractionOf(findParameter(parameters, 'probability'), parameters[0].default),
      toUnit([inputs.zones.centers[0], inputs.zones.centers[1]])[1]
    ],
    resolve(pointer, {inputs, outputs, peek}) {
      const probability = getParameterAt(findParameter(parameters, 'probability'), pointer.unitX);
      const zone = findZoneAt(inputs.zones, pointer.x, pointer.y);
      const quantiles = peek(outputs.quantiles);
      return {
        values: {probability},
        glyphs: [{shape: 'polyline', points: getZoneRing(inputs.zones, zone), closed: true}],
        readout: `probabilities ${probability.toFixed(2)} · zone ${zone} q = ${quantiles && Number.isFinite(quantiles[zone]) ? quantiles[zone].toFixed(1) : 'no points'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getQuadFill(outputs.mesh, inputs.zones, outputs.quantiles, {
      ramp: 'sequential',
      range: [0, 12]
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    },
    {layer: 'dots', context: true, positions: outputs.pointPositions, radius: 1.5}
  ]
});
