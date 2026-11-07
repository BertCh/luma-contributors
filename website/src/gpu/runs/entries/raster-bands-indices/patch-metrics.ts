// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterPatchMetrics on a patch label raster: one area and one perimeter per patch (the bars:
// areas bottom left, perimeters bottom right, one bar per patch label). Labels are drawn behind.
// The patch count rewrites the labels; `countRasterBorder` is a construction prop.

import {GPURasterPatchMetrics} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './patch-metrics.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {createPatchSeeds, getPatchLabels} from './support/patches';

const PATCH_CAPACITY = 41;

export default defineOperationRun({
  parameters,
  rebuildOn: ['countRasterBorder'],
  caption: 'Patch labels in; area and perimeter of every patch out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 48, 1);
    return {...raster, seeds: createPatchSeeds(random, raster.columns, raster.rows, 40)};
  },
  build(builder, inputs, values) {
    const labels = builder.upload('labels', 'uint32', new Uint32Array(inputs.values.length));
    const areas = builder.allocate('areas', 'float32', PATCH_CAPACITY);
    const perimeters = builder.allocate('perimeters', 'float32', PATCH_CAPACITY);
    const pixelCounts = builder.allocate('pixel-counts', 'uint32', PATCH_CAPACITY);
    builder.graph.add(
      new GPURasterPatchMetrics({
        grid: makeGridOfRaster(inputs),
        labels: labels.view,
        countRasterBorder: values.countRasterBorder === 1,
        output: {areas: areas.view, perimeters: perimeters.view, pixelCounts: pixelCounts.view}
      })
    );
    return {
      outputs: {labels, areas, perimeters},
      update: current => labels.write(getPatchLabels(inputs.seeds, current.patchCount))
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.labels, {alpha: 0.85}),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.areas,
      ramp: 'sequential',
      rectangle: [0.03, 0.03, 0.48, 0.3]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.perimeters,
      ramp: 'sequential',
      rectangle: [0.52, 0.03, 0.97, 0.3]
    }
  ]
});
