// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialLagMaximumLikelihood searches rho by maximum likelihood on a spatial-lag response.
// Left: residuals per tract. Right: the estimated coefficients (intercept, slopes and rho last) as
// bars, with the true values as dots.

import {GPUSpatialLagMaximumLikelihood} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {parameters} from './spatial-lag-maximum-likelihood.parameters';
import {findTractUnder, getTractNeighbors, getTractOutline} from './shared/probes';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun<{response: RunBuffer<'float32'>}>({
  parameters,
  caption: 'Maximum-likelihood lag model: estimates (bars) against the truth (dots).',
  process: 'lag',
  rebuildOn: ['maximumPower'],
  addOperation: ({
    builder,
    values,
    weights,
    predictors,
    response,
    table,
    summary,
    status,
    residuals
  }) => {
    builder.graph.add(
      new GPUSpatialLagMaximumLikelihood({
        weights,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        maximumPower: values.maximumPower,
        output: {
          status: {code: status.view},
          table: table.view,
          summary: summary.view,
          residuals: residuals.view
        }
      })
    );
    return {response};
  },
  // The pointer picks a tract: its response is compared with the average of its neighbors' (W y), the
  // lag the model estimates rho from.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const response = peek(outputs.response);
      const estimates = peek(outputs.panel.estimates);
      if (tract < 0 || !response || !estimates) return {};
      const neighbors = getTractNeighbors(inputs.tracts, tract);
      let lagged = 0;
      for (const neighbor of neighbors) lagged += response[neighbor] / neighbors.length;
      const glyphs: RunGlyph[] = [
        getTractOutline(inputs.tracts, tract),
        ...Array.from(neighbors, neighbor => getTractOutline(inputs.tracts, neighbor, 0, true))
      ];
      return {
        focus: tract,
        glyphs,
        readout: `y ${response[tract].toFixed(2)} · Wy ${lagged.toFixed(2)} · rho ${estimates[3].toFixed(2)} (true ${values.strength})`
      };
    }
  }
});
