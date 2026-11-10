// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalCategoricalStatistics behind GPURasterFractionalCoverage: every polygon
// reports how much of each poverty class it covers. Polygons are filled with their majority
// class. The threshold slider shifts the class thresholds of the category raster, so the
// composition (and sometimes the majority) of every polygon changes. The pointer probes a tract:
// its four class shares are bars, labelled by class colour.

import {
  GPURasterFractionalCategoricalStatistics,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './fractional-category-composition.parameters';
import {drawRaster} from './support/bands';
import {
  addFractionalCoverage,
  createZoneInputs,
  drawZones,
  getTractAt,
  getTractRing,
  sampleField,
  uploadZoneMesh
} from './support/zones';
import {readFloat32} from './support/labels';

const CATEGORY_CAPACITY = 4;
// Four ordered poverty classes, low to high.
const CLASS_COLORS = [
  [0.3, 0.6, 1],
  [0.3, 0.78, 0.7],
  [1, 0.75, 0.3],
  [1, 0.35, 0.3]
] as const;
const SHARE_RECTANGLE = [0.03, 0.09, 0.3, 0.3] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['gridWidth'],
  caption:
    'Gridded poverty classes and Chicago tracts in; class shares and majority per tract out.',
  createInputs: ({random, world}) => createZoneInputs(random, world),
  build(builder, inputs, values) {
    const {features} = inputs;
    const result = addFractionalCoverage(builder, inputs, values.gridWidth);
    const poverty = sampleField(inputs, result.columns, result.rows);
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
        proportions: output.proportions,
        shares: builder.allocate('shares', 'float32', CATEGORY_CAPACITY),
        legend: builder.upload(
          'legend',
          'float32x2',
          Float32Array.from({length: CATEGORY_CAPACITY * 2}, (_, index) =>
            index % 2 === 0
              ? inputs.world.width *
                (SHARE_RECTANGLE[0] +
                  (((index >> 1) + 0.5) / CATEGORY_CAPACITY) *
                    (SHARE_RECTANGLE[2] - SHARE_RECTANGLE[0]))
              : inputs.world.height * (SHARE_RECTANGLE[1] - 0.035)
          )
        ),
        legendClasses: builder.upload('legend-classes', 'uint32', Uint32Array.from([0, 1, 2, 3])),
        columns: result.columns,
        rows: result.rows,
        mesh: uploadZoneMesh(builder, features)
      },
      update: current => {
        // Four classes of the gridded poverty measure. The slider moves the middle threshold.
        categories.write(
          Uint32Array.from(poverty, value =>
            value < current.threshold / 2
              ? 0
              : value < current.threshold
                ? 1
                : value < (1 + current.threshold) / 2
                  ? 2
                  : 3
          )
        );
      }
    };
  },
  // A rebuild clears the shares bars; rewrite them from the last probed tract.
  async afterRun({inputs, outputs}) {
    if (inputs.focus.tract < 0) return;
    const proportions = await readFloat32(outputs.proportions);
    const tract = inputs.focus.tract;
    outputs.shares.write(
      proportions.slice(tract * CATEGORY_CAPACITY, (tract + 1) * CATEGORY_CAPACITY)
    );
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a tract',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, peek}) => {
      const tract = getTractAt(inputs.features, pointer.x, pointer.y);
      inputs.focus.tract = tract;
      const proportions = peek(outputs.proportions);
      if (tract < 0 || !proportions) return {readout: 'outside the tracts'};
      const shares = proportions.slice(tract * CATEGORY_CAPACITY, (tract + 1) * CATEGORY_CAPACITY);
      outputs.shares.write(Float32Array.from(shares));
      let majority = 0;
      for (let category = 1; category < CATEGORY_CAPACITY; category++) {
        if (shares[category] > shares[majority]) majority = category;
      }
      return {
        glyphs: [{shape: 'polyline', closed: true, points: getTractRing(inputs.features, tract)}],
        readout: `tract ${inputs.features.geoids[tract].slice(-6)} | ${Math.round(100 * shares[majority])}% class ${majority + 1} | ${[...shares].map(share => Math.round(100 * share)).join('/')}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    drawRaster(outputs, outputs.categories, {context: true}),
    drawZones(inputs.features, outputs.mesh, outputs.majority, {
      ramp: 'categorical',
      categories: CLASS_COLORS,
      alpha: 0.9
    }),
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.8]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.shares,
      range: [0, 1],
      color: [1, 1, 1, 0.9],
      rectangle: SHARE_RECTANGLE
    },
    {
      layer: 'dots',
      positions: outputs.legend,
      values: outputs.legendClasses,
      ramp: 'categorical',
      categories: CLASS_COLORS,
      radius: 5
    }
  ]
});
