// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterZonalStatistics behind GPUPolygonRasterization: polygons become a zone raster, the
// value raster is reduced per zone, and each polygon is filled with its zone statistic. The grid
// width is a compile-time prop; the statistic slider only changes which output
// column is drawn.

import {
  GPURasterZonalStatistics,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './raster-zonal-statistics.parameters';
import {drawRaster} from './support/bands';
import {
  addZoneRaster,
  createZoneInputs,
  drawZones,
  sampleField,
  uploadZoneMesh
} from './support/zones';

const STATISTICS = [
  'means',
  'maximums',
  'minimums',
  'sums',
  'standardDeviations',
  'valueCounts'
] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A value raster and polygons in; one statistic per polygon out.',
  createInputs: ({random, world}) => createZoneInputs(random, world),
  build(builder, inputs, values) {
    const {features} = inputs;
    const zoneRaster = addZoneRaster(builder, inputs, values.gridWidth);
    const {columns, rows} = zoneRaster;
    const cellValues = builder.upload('values', 'float32', sampleField(inputs, columns, rows));
    const zoneCount = features.featureCount;
    const columnsOut = {
      means: builder.allocate('means', 'float32', zoneCount),
      maximums: builder.allocate('maximums', 'float32', zoneCount),
      minimums: builder.allocate('minimums', 'float32', zoneCount),
      sums: builder.allocate('sums', 'float32', zoneCount),
      standardDeviations: builder.allocate('standard-deviations', 'float32', zoneCount),
      valueCounts: builder.allocate('value-counts', 'uint32', zoneCount)
    };
    builder.graph.add(
      new GPURasterZonalStatistics({
        grid: zoneRaster.grid,
        zones: zoneRaster.zones.view,
        values: makeGPURasterBand(cellValues.view, zoneRaster.grid, {id: 'values'}),
        zoneCapacity: zoneCount,
        output: {
          means: columnsOut.means.view,
          maximums: columnsOut.maximums.view,
          minimums: columnsOut.minimums.view,
          sums: columnsOut.sums.view,
          standardDeviations: columnsOut.standardDeviations.view,
          valueCounts: columnsOut.valueCounts.view
        }
      })
    );
    return {
      outputs: {...columnsOut, cellValues, columns, rows, mesh: uploadZoneMesh(builder, features)}
    };
  },
  draw: ({inputs, outputs, values, theme}) => [
    drawRaster(outputs, outputs.cellValues, {
      context: true
    }),
    drawZones(inputs.features, outputs.mesh, outputs[STATISTICS[Math.round(values.statistic)]], {
      ramp: 'sequential'
    }),
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.85]}
  ]
});
