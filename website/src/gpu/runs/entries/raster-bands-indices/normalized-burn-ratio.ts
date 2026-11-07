// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterNBR on synthetic imagery. Vegetation (high NIR, low SWIR2) is strongly positive, bare
// soil sits near zero. `epsilon` and `clamp` are construction props, so the run rebuilds.

import {GPURasterNBR} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {drawRaster, uploadBand} from './support/bands';
import {parameters} from './normalized-burn-ratio.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['epsilon', 'clampMaximum'],
  caption: 'NIR and SWIR2 in; the burn ratio out.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const shortwave = uploadBand(builder, imagery, 'swir2', imagery.shortwaveInfrared);
    const output = builder.allocate('nbr', 'float32', cellCount);
    const outputValidity = builder.allocate('nbr-valid', 'uint32', cellCount);
    new GPURasterNBR({
      width: imagery.columns,
      height: imagery.rows,
      nearInfrared: nearInfrared.band,
      shortwaveInfrared2: shortwave.band,
      epsilon: values.epsilon,
      clamp: [-values.clampMaximum, values.clampMaximum],
      output: output.view,
      outputValidity: outputValidity.view
    }).addToGraph(builder.graph);
    return {outputs: {shortwave: shortwave.buffer, output}};
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.shortwave, {context: true}),
    drawRaster(inputs, outputs.output, {ramp: 'diverging', smoothing: true})
  ]
});
