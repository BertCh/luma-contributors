// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalCategoricalStatistics behind GPURasterFractionalCoverage: every polygon
// reports how much of each land-cover class it covers. Polygons are filled with their majority
// class. The wetness slider shifts the class thresholds of the category raster, so the
// composition (and sometimes the majority) of every polygon changes.

import {
  GPURasterFractionalCategoricalStatistics,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {parameters} from './fractional-category-composition.parameters';
import {drawRaster} from './support/bands';
import {
  addFractionalCoverage,
  createZoneInputs,
  drawZones,
  sampleField,
  uploadZoneMesh
} from './support/zones';

const CATEGORY_CAPACITY = 4;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption: 'A land-cover raster and polygons in; the class mix and majority per polygon out.',
  createInputs: ({random, world}) => ({
    ...createZoneInputs(random, world),
    moisture: createSmoothField(random, 3, 4)
  }),
  build(builder, inputs, values) {
    const {features} = inputs;
    const result = addFractionalCoverage(builder, inputs, values.gridWidth);
    const elevation = sampleField(inputs, result.columns, result.rows);
    const moistureValues = sampleField(
      {...inputs, field: inputs.moisture},
      result.columns,
      result.rows
    );
    const categories = builder.allocate('categories', 'uint32', result.cellCount);
    const zoneCount = features.featureCount;
    const output = {
      coverageSums: builder.allocate('coverage-sums', 'float32', zoneCount * CATEGORY_CAPACITY),
      classAreas: builder.allocate('class-areas', 'float32', zoneCount * CATEGORY_CAPACITY),
      proportions: builder.allocate('proportions', 'float32', zoneCount * CATEGORY_CAPACITY),
      coveredArea: builder.allocate('covered-area', 'float32', zoneCount),
      validArea: builder.allocate('valid-area', 'float32', zoneCount),
      missingArea: builder.allocate('missing-area', 'float32', zoneCount),
      majority: builder.allocate('majority', 'uint32', zoneCount),
      minority: builder.allocate('minority', 'uint32', zoneCount),
      validity: builder.allocate('validity', 'uint32', zoneCount)
    };
    builder.graph.add(
      new GPURasterFractionalCategoricalStatistics({
        grid: result.grid,
        input: makeGPURasterBand(categories.view, result.grid, {id: 'categories'}),
        zoneCount,
        categoryCapacity: CATEGORY_CAPACITY,
        coverage: result.output,
        output: Object.fromEntries(
          Object.entries(output).map(([name, buffer]) => [name, buffer.view])
        ) as never
      })
    );
    return {
      outputs: {
        categories,
        majority: output.majority,
        columns: result.columns,
        rows: result.rows,
        mesh: uploadZoneMesh(builder, features)
      },
      update: current => {
        // 0 water (low), 1 bare, 2 wet vegetation, 3 dry vegetation.
        categories.write(
          Uint32Array.from(elevation, (height, index) =>
            height < current.wetness * 0.8
              ? 0
              : moistureValues[index] > current.wetness
                ? 2
                : height > 0.7
                  ? 3
                  : 1
          )
        );
      }
    };
  },
  draw: ({inputs, outputs, theme}) => [
    drawRaster(outputs, outputs.categories, {context: true}),
    drawZones(inputs.features, outputs.mesh, outputs.majority),
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.8]}
  ]
});
