// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeodesicPairs measures many origin-target pairs on the globe. Lines join each pair on an
// equirectangular map; the dots are the geodesic midpoints the operation wrote, colored by the
// pair's distance in meters.

import {GPUGeodesicPairs} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {addEquirectangular, addPairSegments} from './shared/shape-segments';
import {parameters} from './geodesic-pairs.parameters';

const PAIR_COUNT = 30;

export default defineOperationRun({
  parameters,
  caption: 'Distance and midpoint of each pair, measured on the ellipsoid.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, inputs, values) {
    const random = makeRandom(inputs.seed);
    const origins = new Float32Array(PAIR_COUNT * 2);
    const targets = new Float32Array(PAIR_COUNT * 2);
    for (let pair = 0; pair < PAIR_COUNT; pair++) {
      const longitude = -140 + random() * 280;
      const latitude = -55 + random() * 110;
      origins[pair * 2] = longitude;
      origins[pair * 2 + 1] = latitude;
      targets[pair * 2] = longitude + (random() - 0.5) * 80 * values.spread * 2;
      targets[pair * 2 + 1] = Math.max(
        -80,
        Math.min(80, latitude + (random() - 0.5) * 50 * values.spread * 2)
      );
    }
    const originBuffer = builder.upload('origins', 'float32x2', origins);
    const targetBuffer = builder.upload('targets', 'float32x2', targets);
    const distances = builder.allocate('distances', 'float32', PAIR_COUNT);
    const midpoints = builder.allocate('midpoints', 'float32x2', PAIR_COUNT);
    builder.graph.add(
      new GPUGeodesicPairs({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          units: 'meters'
        },
        origins: originBuffer.view,
        targets: targetBuffer.view,
        output: {distances: distances.view, midpoints: midpoints.view}
      })
    );
    const mapOrigins = addEquirectangular(builder, 'origins', originBuffer, inputs.world);
    return {
      outputs: {
        pairs: addPairSegments(
          builder,
          'pairs',
          mapOrigins,
          addEquirectangular(builder, 'targets', targetBuffer, inputs.world)
        ),
        mapOrigins,
        midpoints: addEquirectangular(builder, 'midpoints', midpoints, inputs.world),
        distances
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.pairs},
    {layer: 'dots', positions: outputs.mapOrigins, context: true, radius: 3},
    {layer: 'dots', positions: outputs.midpoints, values: outputs.distances, radius: 11}
  ]
});
