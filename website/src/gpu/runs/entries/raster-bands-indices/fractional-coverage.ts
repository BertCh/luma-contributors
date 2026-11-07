// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalCoverage: for every polygon and every grid cell it touches, the exact share
// of the cell the polygon covers. The output is a compact list of (zone, cell, fraction) tuples;
// bars show the fractions in tuple order, so each polygon is a run of full cells (1.0) between
// partial edge cells.

import {defineOperationRun} from '../../define-run';
import {parameters} from './fractional-coverage.parameters';
import {addFractionalCoverage, createZoneInputs, uploadZoneMesh} from './support/zones';

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'Polygons and a grid in; one (zone, cell, covered fraction) tuple per overlap out.',
  createInputs: ({random, world}) => createZoneInputs(random, world),
  build(builder, inputs, values) {
    const {features} = inputs;
    const result = addFractionalCoverage(builder, inputs, values.gridWidth);
    return {
      outputs: {
        fractions: result.coverage.fractions,
        tupleCount: result.coverage.count,
        mesh: uploadZoneMesh(builder, features)
      }
    };
  },
  draw: ({inputs, outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.5]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.fractions,
      ramp: 'sequential',
      range: [0, 1]
    }
  ]
});
