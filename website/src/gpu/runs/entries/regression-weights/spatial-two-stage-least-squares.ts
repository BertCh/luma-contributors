// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialTwoStageLeastSquares on a spatial-lag response. Left: structural residuals per tract.
// Right: the coefficient table's estimates (intercept, two slopes and the spatial lag rho) as bars,
// with the true values as dots.

import {GPUSpatialTwoStageLeastSquares} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunGlyph} from '../../interaction';
import {parameters} from './spatial-two-stage-least-squares.parameters';
import {findTractUnder, getTractNeighbors, getTractOutline} from './shared/probes';
import {defineTableModelRun} from './shared/table-model';

export default defineTableModelRun({
  parameters,
  caption: 'Spatial 2SLS estimates (bars) against the true coefficients (dots).',
  process: 'lag',
  rebuildOn: ['instrumentOrder'],
  // The pointer picks a tract: its first-order W neighbors (solid) and, for instrumentOrder 2, their
  // neighbors (dashed) are the tracts whose values build its instruments.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const estimates = peek(outputs.panel.estimates);
      if (tract < 0 || !estimates) return {};
      const first = new Set(getTractNeighbors(inputs.tracts, tract));
      const second = new Set<number>();
      if (values.instrumentOrder === 2) {
        for (const neighbor of first) {
          for (const other of getTractNeighbors(inputs.tracts, neighbor)) {
            if (other !== tract && !first.has(other)) second.add(other);
          }
        }
      }
      const glyphs: RunGlyph[] = [
        getTractOutline(inputs.tracts, tract),
        ...Array.from(first, neighbor => getTractOutline(inputs.tracts, neighbor)),
        ...Array.from(second, other => getTractOutline(inputs.tracts, other, 0, true))
      ];
      return {
        focus: tract,
        glyphs,
        readout: `tract ${tract} · ${first.size} + ${second.size} instrument tracts · rho ${estimates[3].toFixed(2)}`
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
      new GPUSpatialTwoStageLeastSquares({
        weights,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        instrumentOrder: values.instrumentOrder === 1 ? 1 : 2,
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
