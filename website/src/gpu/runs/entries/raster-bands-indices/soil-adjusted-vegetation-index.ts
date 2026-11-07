// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSAVI on synthetic imagery. `soilAdjustment` (L) is a construction prop: zero gives
// NDVI, one flattens the soil response. The run rebuilds when it moves.

import {GPURasterSAVI} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {drawRaster, uploadBand} from './support/bands';
import {parameters} from './soil-adjusted-vegetation-index.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['soilAdjustment', 'epsilon'],
  caption: 'NIR and red in; SAVI out, with the soil-brightness correction L.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const red = uploadBand(builder, imagery, 'red', imagery.red);
    const output = builder.allocate('savi', 'float32', cellCount);
    const outputValidity = builder.allocate('savi-valid', 'uint32', cellCount);
    new GPURasterSAVI({
      width: imagery.columns,
      height: imagery.rows,
      nearInfrared: nearInfrared.band,
      red: red.band,
      soilAdjustment: values.soilAdjustment,
      epsilon: values.epsilon,
      output: output.view,
      outputValidity: outputValidity.view
    }).addToGraph(builder.graph);
    return {outputs: {red: red.buffer, output}};
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.red, {context: true}),
    drawRaster(inputs, outputs.output, {ramp: 'sequential', range: [0, 1], smoothing: true})
  ]
});
