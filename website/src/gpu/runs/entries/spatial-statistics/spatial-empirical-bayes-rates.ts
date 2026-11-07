// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialEmpiricalBayesRates: shrinks each tract's raw rate toward the rate of its neighbors
// instead of the overall rate, so local risk differences survive. Colors are the spatially
// smoothed rates; the dots are the raw rates.

import {GPUSpatialEmpiricalBayesRates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {createRateSeeds, getRateCounts, uploadTractMesh} from './shared/tract-rates';
import {parameters} from './spatial-empirical-bayes-rates.parameters';

export default defineOperationRun({
  parameters,
  caption: "Rates smoothed toward each tract's neighbors, with the raw rates as dots.",
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {tracts, seeds: createRateSeeds(random, tracts.count)};
  },
  build(builder, {tracts, seeds}) {
    const events = builder.allocate('events', 'float32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const spatialRates = builder.allocate('spatial-rates', 'float32', tracts.count);
    const smoothedRates = builder.allocate('smoothed-rates', 'float32', tracts.count);
    const rawRates = builder.allocate('raw-rates', 'float32', tracts.count);
    builder.graph.add(
      new GPUSpatialEmpiricalBayesRates({
        events: events.view,
        populations: populations.view,
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        output: {spatialRates: spatialRates.view, smoothedRates: smoothedRates.view}
      })
    );
    return {
      outputs: {spatialRates, rawRates, ...uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(
          seeds,
          tracts.values,
          current.riskContrast,
          current.populationScale
        );
        events.write(counts.events);
        populations.write(counts.populations);
        rawRates.write(
          Float32Array.from(counts.events, (count, index) => count / counts.populations[index])
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.spatialRates,
      triangleCount: inputs.tracts.triangleCount,
      range: [0.015, 0.065]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.rawRates,
      range: [0.015, 0.065],
      radius: 4
    }
  ]
});
