// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterWatershedSegmentation: basins flood upward from marker cells (the lowest local minima of
// the surface) and meet along ridges. Each basin gets the label of its marker. The number of
// markers rewrites the marker raster; method and connectivity are construction props.

import {GPURasterWatershedSegmentation} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './watershed-segmentation.parameters';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';
import {findLocalMinima} from './support/minima';

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'connectivity'],
  caption: 'A surface and basin markers in; the basin of every cell out.',
  createInputs: ({random, world}) => {
    const raster = createFractalRaster(random, world, 54, 5);
    return {...raster, minima: findLocalMinima(raster)};
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const surface = uploadBand(builder, inputs, 'surface', inputs.values);
    const markers = builder.allocate('markers', 'uint32', cellCount);
    const labels = builder.allocate('labels', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    builder.graph.add(
      new GPURasterWatershedSegmentation({
        grid: makeGridOfRaster(inputs),
        input: surface.band,
        markers: markers.view,
        connectivity: values.connectivity === 1 ? 8 : 4,
        method: values.method === 1 ? 'minimum-spanning-forest' : 'fifo',
        output: {labels: labels.view, status: {converged: converged.view}}
      })
    );
    return {
      outputs: {surface: surface.buffer, labels},
      update: current => {
        const data = new Uint32Array(cellCount);
        inputs.minima.slice(0, current.markerCount).forEach((cell, index) => {
          data[cell] = index + 1;
        });
        markers.write(data);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.surface, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.labels, {alpha: 0.75, hillshade: 0.4})
  ]
});
