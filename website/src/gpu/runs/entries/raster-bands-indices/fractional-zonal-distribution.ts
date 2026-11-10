// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFractionalZonalDistribution behind GPURasterFractionalCoverage: the coverage-weighted
// quantile of the raster values inside every polygon (filled). The probability is a per-frame input,
// set by the pointer; the quantile method is compile-time.

import {
  GPURasterFractionalZonalDistribution,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {bindAxes} from '../../interaction';
import {defineOperationRun} from '../../define-run';
import {parameters} from './fractional-zonal-distribution.parameters';
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

const PAD = bindAxes(parameters, {kind: 'level', hint: 'move the probability', x: 'probability'});

export default defineOperationRun({
  parameters,
  rebuildOn: ['quantileMethod', 'gridWidth'],
  caption: 'Polygons over a raster: the quantile of the values each polygon covers.',
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
    const probabilities = builder.allocate('probabilities', 'float32', 1);
    const output = {
      means: builder.allocate('means', 'float32', zoneCount),
      populationVariances: builder.allocate('population-variances', 'float32', zoneCount),
      sampleVariances: builder.allocate('sample-variances', 'float32', zoneCount),
      effectiveSampleSizes: builder.allocate('effective-sample-sizes', 'float32', zoneCount),
      weightSums: builder.allocate('weight-sums', 'float32', zoneCount),
      squaredWeightSums: builder.allocate('squared-weight-sums', 'float32', zoneCount),
      coveredArea: builder.allocate('covered-area', 'float32', zoneCount),
      validArea: builder.allocate('valid-area', 'float32', zoneCount),
      quantiles: builder.allocate('quantiles', 'float32', zoneCount),
      quantileValidity: builder.allocate('quantile-validity', 'uint32', zoneCount),
      validity: builder.allocate('validity', 'uint32', zoneCount),
      status: builder.allocate('status', 'uint32', zoneCount)
    };
    builder.graph.add(
      new GPURasterFractionalZonalDistribution({
        grid: result.grid,
        input: makeGPURasterBand(cellValues.view, result.grid, {id: 'values'}),
        zoneCount,
        coverage: result.output,
        probabilities: probabilities.view,
        quantileMethod: values.quantileMethod === 1 ? 'inverse-cdf' : 'exactextract',
        output: Object.fromEntries(
          Object.entries(output).map(([name, buffer]) => [name, buffer.view])
        ) as never
      })
    );
    return {
      outputs: {
        quantiles: output.quantiles,
        sampleVariances: output.sampleVariances,
        cellValues,
        columns: result.columns,
        rows: result.rows,
        mesh: uploadZoneMesh(builder, features)
      },
      update: current => probabilities.write(Float32Array.of(current.probability))
    };
  },
  // The pointer is the probability (x), shown on a scale along the bottom; the readout gives the
  // quantile of the tract under it.
  interaction: {
    ...PAD,
    resolve: (pointer, state) => {
      const base = PAD.resolve(pointer, state);
      const current = {...state.values, ...base.values};
      const {inputs, outputs, world, peek} = state;
      const tract = getTractAt(inputs.features, pointer.x, pointer.y);
      const quantiles = peek(outputs.quantiles);
      const left = world.width * 0.05;
      const right = world.width * 0.95;
      const y = world.height * 0.04;
      const x = left + (right - left) * current.probability;
      return {
        ...base,
        glyphs: [
          {shape: 'line', from: [left, y], to: [right, y]},
          {shape: 'dot', at: [x, y]},
          ...(tract >= 0
            ? ([
                {shape: 'polyline', closed: true, points: getTractRing(inputs.features, tract)}
              ] as const)
            : [])
        ],
        readout: `probability ${current.probability.toFixed(2)} | tract quantile ${tract >= 0 && quantiles ? quantiles[tract].toFixed(2) : '-'}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    drawRaster(outputs, outputs.cellValues, {context: true}),
    drawZones(inputs.features, outputs.mesh, outputs.quantiles, {
      ramp: 'sequential',
      range: [0.1, 0.9]
    }),
    {layer: 'lines', context: true, segments: outputs.mesh.outline, color: [...theme.ink, 0.8]}
  ]
});
