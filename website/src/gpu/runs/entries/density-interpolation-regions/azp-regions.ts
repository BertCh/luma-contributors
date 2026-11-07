// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAZPRegions improves a starting partition by moving border tracts between regions while the
// within-region variance falls. With one iteration the picture is close to the starting partition (a
// Voronoi split of random seed tracts); more iterations straighten the borders onto the data.

import {
  getGPUAZPRegionsParameterValues,
  GPU_AZP_STATUS,
  GPUAZPRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './azp-regions.parameters';
import {uploadTractMesh} from './support/geometry';
import {createRegionScene, drawLabels} from './support/regions';

export default defineOperationRun({
  parameters,
  caption: 'Starting partition improved by moving border tracts.',
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 8, 2),
  build(builder, scene, current) {
    const {tracts, attributes, columnCount} = scene;
    // Starting partition: every tract joins its nearest seed tract, so regions start contiguous.
    const random = makeRandom(current.seed);
    const seeds = Array.from({length: current.regionCapacity}, () =>
      Math.floor(random() * tracts.count)
    );
    const initialLabels = Uint32Array.from({length: tracts.count}, (_, tract) => {
      let best = 0;
      let bestDistance = Infinity;
      seeds.forEach((seed, index) => {
        const distance = Math.hypot(
          tracts.centers[tract * 2] - tracts.centers[seed * 2],
          tracts.centers[tract * 2 + 1] - tracts.centers[seed * 2 + 1]
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      return best;
    });
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUAZPRegions({
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('attributes', 'float32', attributes).view,
        columnCount,
        standardize: true,
        initialLabels: builder.upload('initial-labels', 'uint32', initialLabels).view,
        regionCapacity: current.regionCapacity,
        maximumIterations: current.maximumIterations,
        parameters: builder.upload(
          'parameters',
          'uint32',
          getGPUAZPRegionsParameterValues({seed: current.seed})
        ).view,
        output: {
          labels: labels.view,
          objective: builder.allocate('objective', 'float32', 1).view,
          status: builder.allocate('status', 'uint32', GPU_AZP_STATUS.length).view
        }
      })
    );
    return {outputs: {labels, ...uploadTractMesh(builder, tracts)}};
  },
  draw: ({inputs, outputs}) => drawLabels(inputs, outputs, outputs.labels)
});
