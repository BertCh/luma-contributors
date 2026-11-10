// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialErrorGM estimates the spatial error parameter by generalized moments. Left: residuals
// per tract. Right: the estimated coefficients (intercept, slopes, lambda last) as bars, with the
// true values as dots.

import {GPUSpatialErrorGM} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunGlyph} from '../../interaction';
import {parameters} from './spatial-error-gm.parameters';
import {findTractUnder, getTractNeighbors, getTractOutline} from './shared/probes';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun({
  parameters,
  caption: 'Generalized-moments error model: estimates (bars) against the truth (dots).',
  process: 'error',
  rebuildOn: ['estimator'],
  // The pointer picks a tract: its residual is compared with the mean of its W neighbors, the spatial
  // correlation the error model removes.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const residuals = peek(outputs.residuals);
      const estimates = peek(outputs.panel.estimates);
      if (tract < 0 || !residuals || !estimates) return {};
      const neighbors = getTractNeighbors(inputs.tracts, tract);
      let neighborMean = 0;
      for (const neighbor of neighbors) neighborMean += residuals[neighbor] / neighbors.length;
      const glyphs: RunGlyph[] = [
        getTractOutline(inputs.tracts, tract),
        ...Array.from(neighbors, neighbor => getTractOutline(inputs.tracts, neighbor, 0, true))
      ];
      return {
        focus: tract,
        glyphs,
        readout: `residual ${residuals[tract].toFixed(2)} · neighbors ${neighborMean.toFixed(2)} · lambda ${estimates[3].toFixed(2)} (true ${values.strength})`
      };
    }
  },
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
      new GPUSpatialErrorGM({
        weights,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        estimator: values.estimator === 0 ? 'homoskedastic' : 'kelejian-prucha',
        output: {
          status: {code: status.view},
          table: table.view,
          summary: summary.view,
          residuals: residuals.view
        }
      })
    );
  }
});
