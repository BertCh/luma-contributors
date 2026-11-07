// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAssuncaoReisStandardization: turns raw rates into standardized rates (z-like scores) that
// account for each tract's population, so small tracts do not dominate the extremes. Dots show
// the raw rate; the polygon colors are the standardized rates (red above, blue below the mean).

import {GPUAssuncaoReisStandardization} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {createRateSeeds, getRateCounts, uploadTractMesh} from './shared/tract-rates';
import {parameters} from './assuncao-reis-standardization.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Population-aware standardized rates, with the raw rates as dots.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {tracts, seeds: createRateSeeds(random, tracts.count)};
  },
  build(builder, {tracts, seeds}) {
    const events = builder.allocate('events', 'float32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const rawRates = builder.allocate('raw-rates', 'float32', tracts.count);
    const standardizedRates = builder.allocate('standardized-rates', 'float32', tracts.count);
    builder.graph.add(
      new GPUAssuncaoReisStandardization({
        events: events.view,
        populations: populations.view,
        output: {standardizedRates: standardizedRates.view}
      })
    );
    return {
      outputs: {standardizedRates, rawRates, ...uploadTractMesh(builder, tracts)},
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
      values: outputs.standardizedRates,
      triangleCount: inputs.tracts.triangleCount,
      ramp: 'diverging',
      range: [-3, 3]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      positions: outputs.centers,
      values: outputs.rawRates,
      range: [0, 0.08],
      radius: 3.5,
      context: true
    }
  ]
});
