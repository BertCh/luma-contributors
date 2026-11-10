// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEmpiricalBayesRates: shrinks noisy raw rates toward the overall rate, small populations most.
// The pointer is a curtain: the raw rates on its left, the smoothed rates on its right, on one
// scale, so the extremes of the small tracts visibly shrink toward the overall rate.

import {GPUEmpiricalBayesRates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {getRateCounts, pickTract, uploadTractMesh} from './shared/tract-rates';
import {parameters} from './empirical-bayes-rates.parameters';

export default defineOperationRun({
  parameters,
  caption:
    'Raw and smoothed shares of residents age 65+ (Census counts with simulated sampling noise).',
  createInputs: ({random, world}) => ({tracts: createTracts(random, world), focus: {row: -1}}),
  build(builder, {tracts}) {
    const events = builder.allocate('events', 'float32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const smoothedRates = builder.allocate('smoothed-rates', 'float32', tracts.count);
    const rawRates = builder.allocate('raw-rates', 'float32', tracts.count);
    builder.graph.add(
      new GPUEmpiricalBayesRates({
        events: events.view,
        populations: populations.view,
        output: {smoothedRates: smoothedRates.view, rawRates: rawRates.view}
      })
    );
    return {
      outputs: {smoothedRates, rawRates, ...uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(tracts, current.riskContrast, current.populationScale);
        events.write(counts.events);
        populations.write(counts.populations);
      }
    };
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe raw and smoothed',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const row = pickTract(inputs.tracts, pointer, inputs.focus.row);
      inputs.focus.row = row;
      const raw = peek(outputs.rawRates)?.[row];
      const smoothed = peek(outputs.smoothedRates)?.[row];
      const rates =
        row >= 0 && raw !== undefined && smoothed !== undefined
          ? ` · tract ${(raw * 100).toFixed(0)}% -> ${(smoothed * 100).toFixed(0)}%`
          : '';
      return {
        curtain: pointer.x,
        glyphs: [{shape: 'curtain', x: pointer.x}],
        readout: `raw | smoothed${rates}`
      };
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      {
        layer: 'polygons',
        positions: outputs.vertices,
        indices: outputs.meshIndices,
        owners: outputs.owners,
        values: outputs.rawRates,
        triangleCount: inputs.tracts.triangleCount,
        range: [0, 0.35],
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'polygons',
        positions: outputs.vertices,
        indices: outputs.meshIndices,
        owners: outputs.owners,
        values: outputs.smoothedRates,
        triangleCount: inputs.tracts.triangleCount,
        range: [0, 0.35],
        clip: {side: 'right', x: curtain}
      },
      {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount}
    ];
  }
});
