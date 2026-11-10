// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterZonalStatistics behind GPUPolygonRasterization: polygons become a zone raster, the
// value raster is reduced per zone, and each polygon is filled with its zone statistic. The grid
// width is a compile-time prop; the statistic slider only changes which output
// column is drawn. The pointer probes a tract: the raster cells assigned to its zone (cell centers
// inside the polygon) are highlighted over the grid.

import {
  GPURasterZonalStatistics,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './raster-zonal-statistics.parameters';
import {drawRaster} from './support/bands';
import {HIGHLIGHT_CATEGORIES, readUint32} from './support/labels';
import {
  addZoneRaster,
  createGridLines,
  createZoneInputs,
  drawZones,
  getTractAt,
  getTractRing,
  sampleField,
  uploadZoneMesh
} from './support/zones';

const RANGES = {
  means: [0, 1],
  maximums: [0, 1],
  minimums: [0, 1],
  sums: [0, 40],
  standardDeviations: [0, 0.25],
  valueCounts: [0, 200]
} as const;

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
      outputs: {
        ...columnsOut,
        cellValues,
        zones: zoneRaster.zones,
        hover: builder.allocate('hover', 'uint32', zoneRaster.cellCount),
        grid: builder.upload(
          'grid-lines',
          'float32x4',
          createGridLines(inputs.world, columns, rows)
        ),
        columns,
        rows,
        mesh: uploadZoneMesh(builder, features)
      }
    };
  },
  // A rebuild clears the highlight buffer; rewrite it from the last probed tract.
  async afterRun({inputs, outputs}) {
    if (inputs.focus.tract < 0) return;
    const zones = await readUint32(outputs.zones);
    outputs.hover.write(Uint32Array.from(zones, zone => (zone === inputs.focus.tract ? 1 : 0)));
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a tract',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, peek}) => {
      const tract = getTractAt(inputs.features, pointer.x, pointer.y);
      inputs.focus.tract = tract;
      const zones = peek(outputs.zones);
      if (tract < 0 || !zones) return {readout: 'outside the tracts'};
      outputs.hover.write(Uint32Array.from(zones, zone => (zone === tract ? 1 : 0)));
      const counts = peek(outputs.valueCounts);
      const means = peek(outputs.means);
      const deviations = peek(outputs.standardDeviations);
      return {
        glyphs: [{shape: 'polyline', closed: true, points: getTractRing(inputs.features, tract)}],
        readout: `tract ${inputs.features.geoids[tract].slice(-6)} | ${counts?.[tract] ?? '-'} cells | mean ${means ? means[tract].toFixed(2) : '-'} | sd ${deviations ? deviations[tract].toFixed(2) : '-'}`
      };
    }
  },
  draw: ({inputs, outputs, values, theme}) => {
    const statistic = STATISTICS[Math.round(values.statistic)];
    return [
      drawRaster(outputs, outputs.cellValues, {context: true}),
      drawZones(inputs.features, outputs.mesh, outputs[statistic], {
        ramp: 'sequential',
        range: RANGES[statistic],
        alpha: 0.75
      }),
      {layer: 'lines', context: true, segments: outputs.grid, color: [...theme.ink, 0.25]},
      {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.85]},
      drawRaster(outputs, outputs.hover, {ramp: 'categorical', categories: HIGHLIGHT_CATEGORIES})
    ];
  }
});
