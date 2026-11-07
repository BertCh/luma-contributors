// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRegionGraph on a segmentation: region adjacency (shared boundary faces) and the mean
// difference across every adjacent pair, with regions closer than `mergeThreshold` merged. Dots mark
// the regions and take the color of the region they were merged into (the mapping output);
// regions with close means (`mergeThreshold`) end up sharing a color.

import {GPURasterRegionGraph} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './region-adjacency-graph.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {createPatchSeeds, getPatchLabels} from './support/patches';

const REGION_COUNT = 24;

export default defineOperationRun({
  parameters,
  rebuildOn: ['mergeThreshold', 'mergeMode', 'maximumIterations'],
  caption: 'A segmentation and a value raster in; region merges out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 48, 1);
    const seeds = createPatchSeeds(random, raster.columns, raster.rows, REGION_COUNT);
    const labels = getPatchLabels(seeds, REGION_COUNT).map(label => label - 1);
    const regionMeans = Float32Array.from({length: REGION_COUNT}, () => random());
    const values = Float32Array.from(labels, label => regionMeans[label] + (random() - 0.5) * 0.04);
    const centers = Float32Array.from({length: REGION_COUNT * 2}, (_, index) =>
      index % 2 === 0
        ? ((seeds.seeds[(index >> 1) * 3] + 0.5) / raster.columns) * world.width
        : world.height * (1 - (seeds.seeds[(index >> 1) * 3 + 1] + 0.5) / raster.rows)
    );
    return {...raster, labels, values, centers};
  },
  build(builder, inputs, values) {
    const labels = builder.upload('labels', 'uint32', inputs.labels);
    const cellValues = builder.upload('values', 'float32', inputs.values);
    const adjacency = builder.allocate('adjacency', 'uint32', REGION_COUNT ** 2);
    const weights = builder.allocate('weights', 'float32', REGION_COUNT ** 2);
    const mapping = builder.allocate('mapping', 'uint32', REGION_COUNT);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPURasterRegionGraph({
        grid: makeGridOfRaster(inputs),
        labels: labels.view,
        values: cellValues.view,
        regionCount: REGION_COUNT,
        mergeMode: values.mergeMode === 1 ? 'adaptive' : 'fixed',
        mergeThreshold: values.mergeThreshold,
        maximumIterations: values.maximumIterations,
        output: {
          adjacency: adjacency.view,
          weights: weights.view,
          mapping: mapping.view,
          status: status.view
        }
      })
    );
    const centers = builder.upload('centers', 'float32x2', inputs.centers);
    return {outputs: {labels, weights, mapping, adjacency, centers}};
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.labels, {alpha: 0.4}),
    {layer: 'dots', positions: outputs.centers, values: outputs.mapping, radius: 9}
  ]
});
