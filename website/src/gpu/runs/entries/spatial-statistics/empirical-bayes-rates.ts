// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEmpiricalBayesRates: shrinks noisy raw rates toward the overall rate, small populations most.
// The tract colors are the smoothed rates; the dots are the raw rates on the same scale.

import {GPUEmpiricalBayesRates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {createRateSeeds, getRateCounts, uploadTractMesh} from './shared/tract-rates';
import {parameters} from './empirical-bayes-rates.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Smoothed rates per tract, with the raw rates as dots.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {tracts, seeds: createRateSeeds(random, tracts.count)};
  },
  build(builder, {tracts, seeds}, values) {
    const events = builder.allocate('events', 'float32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const smoothedRates = builder.allocate('smoothed-rates', 'float32', tracts.count);
    const rawRates = builder.allocate('raw-rates', 'float32', tracts.count);
    builder.graph.add(
      new GPUEmpiricalBayesRates({
        events: events.view,
        populations: populations.view,
        clampPriorVariance: values.clampPriorVariance === 1,
        output: {smoothedRates: smoothedRates.view, rawRates: rawRates.view}
      })
    );
    return {
      outputs: {smoothedRates, rawRates, ...uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(
          seeds,
          tracts.values,
          current.riskContrast,
          current.populationScale
        );
        events.write(counts.events);
        populations.write(counts.populations);
      }
    };
  },
  rebuildOn: ['clampPriorVariance'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.smoothedRates,
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
