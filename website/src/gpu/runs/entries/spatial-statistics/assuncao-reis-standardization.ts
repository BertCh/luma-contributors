// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAssuncaoReisStandardization: turns raw rates into standardized rates (z-like scores) that
// account for each tract's population, so small tracts do not dominate the extremes. Dots show
// the raw rate of the hovered tract; the polygon colors are the standardized rates (red above,
// blue below the mean). Hovering a tract outlines it and reads its counts, raw rate and score.

import {GPUAssuncaoReisStandardization} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {createTractOutline, getRateCounts, pickTract, uploadTractMesh} from './shared/tract-rates';
import {parameters} from './assuncao-reis-standardization.parameters';

export default defineOperationRun({
  parameters,
  caption:
    'Standardized shares of residents age 65+ (Census counts with simulated sampling noise).',
  createInputs: ({random, world}) => ({tracts: createTracts(random, world), focus: {row: -1}}),
  build(builder, {tracts}) {
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
        const counts = getRateCounts(tracts, current.riskContrast, current.populationScale);
        events.write(counts.events);
        populations.write(counts.populations);
        rawRates.write(
          Float32Array.from(counts.events, (count, index) => count / counts.populations[index])
        );
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = pickTract(inputs.tracts, pointer, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const counts = getRateCounts(inputs.tracts, values.riskContrast, values.populationScale);
      const score = peek(outputs.standardizedRates)?.[row];
      const rate = counts.events[row] / counts.populations[row];
      return {
        glyphs: [
          createTractOutline(inputs.tracts, row),
          {
            shape: 'crosshair',
            at: [inputs.tracts.centers[row * 2], inputs.tracts.centers[row * 2 + 1]],
            size: 10
          }
        ],
        readout: `${counts.events[row]}/${counts.populations[row]} · raw ${(rate * 100).toFixed(1)}% · z ${
          score === undefined ? '…' : (score >= 0 ? '+' : '') + score.toFixed(1)
        }`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.meshIndices,
      owners: outputs.owners,
      values: outputs.standardizedRates,
      triangleCount: inputs.tracts.triangleCount,
      ramp: 'diverging',
      range: [-3, 3]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
  ]
});
