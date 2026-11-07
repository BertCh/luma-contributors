// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEVI on synthetic imagery. The blue band carries atmospheric haze; EVI subtracts it
// (the 7.5 blue term), so raising the haze slider changes the index. Haze only rewrites a band.

import {GPURasterEVI} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {drawRaster, uploadBand} from './support/bands';
import {parameters} from './enhanced-vegetation-index.parameters';

export default defineOperationRun({
  parameters,
  caption: 'NIR, red and blue in; the enhanced vegetation index out.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery) {
    const cellCount = imagery.values.length;
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const red = uploadBand(builder, imagery, 'red', imagery.red);
    const blue = uploadBand(builder, imagery, 'blue', imagery.green);
    const output = builder.allocate('evi', 'float32', cellCount);
    const outputValidity = builder.allocate('evi-valid', 'uint32', cellCount);
    new GPURasterEVI({
      width: imagery.columns,
      height: imagery.rows,
      nearInfrared: nearInfrared.band,
      red: red.band,
      blue: blue.band,
      output: output.view,
      outputValidity: outputValidity.view
    }).addToGraph(builder.graph);
    return {
      outputs: {nearInfrared: nearInfrared.buffer, output},
      update: current => blue.buffer.write(imagery.green.map(value => value * 0.5 + current.haze))
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.nearInfrared, {context: true}),
    drawRaster(inputs, outputs.output, {ramp: 'sequential', smoothing: true})
  ]
});
