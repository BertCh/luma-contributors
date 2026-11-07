// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalZonalStatistics behind GPURasterFractionalCoverage: each polygon aggregates
// the raster cells it overlaps, weighted by the exact covered fraction of every cell
// (extensive quantity, so totals scale with coverage).

import {
  GPURasterFractionalZonalStatistics,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './fractional-zonal-statistics.parameters';
import {drawRaster} from './support/bands';
import {
  addFractionalCoverage,
  createZoneInputs,
  drawZones,
  sampleField,
  uploadZoneMesh
} from './support/zones';

const STATISTICS = ['means', 'totals', 'coveredArea', 'validArea'] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A value raster and polygons in; coverage-weighted statistics per polygon out.',
  createInputs: ({random, world}) => createZoneInputs(random, world),
  build(builder, inputs, values) {
    const {features} = inputs;
    const result = addFractionalCoverage(builder, inputs, values.gridWidth);
    const cellValues = builder.upload(
      'values',
      'float32',
      sampleField(inputs, result.columns, result.rows)
    );
    const zoneCount = features.featureCount;
    const output = {
      totals: builder.allocate('totals', 'float32', zoneCount),
      means: builder.allocate('means', 'float32', zoneCount),
      coveredArea: builder.allocate('covered-area', 'float32', zoneCount),
      validArea: builder.allocate('valid-area', 'float32', zoneCount),
      validity: builder.allocate('validity', 'uint32', zoneCount)
    };
    builder.graph.add(
      new GPURasterFractionalZonalStatistics({
        grid: result.grid,
        input: makeGPURasterBand(cellValues.view, result.grid, {id: 'values'}),
        zoneCount,
        coverage: result.output,
        quantity: 'extensive',
        output: {
          totals: output.totals.view,
          means: output.means.view,
          coveredArea: output.coveredArea.view,
          validArea: output.validArea.view,
          validity: output.validity.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        cellValues,
        columns: result.columns,
        rows: result.rows,
        mesh: uploadZoneMesh(builder, features)
      }
    };
  },
  draw: ({inputs, outputs, values, theme}) => [
    drawRaster(outputs, outputs.cellValues, {context: true}),
    drawZones(inputs.features, outputs.mesh, outputs[STATISTICS[Math.round(values.statistic)]], {
      ramp: 'sequential'
    }),
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.8]}
  ]
});
