// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterNDWI on synthetic imagery: water reads high. `method` picks the McFeeters (green, NIR)
// or Gao (NIR, SWIR) recipe; both are compile-time props, so the run rebuilds on change.

import {GPURasterNDWI} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {drawRaster, uploadBand} from './support/bands';
import {parameters} from './normalized-difference-water-index.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'epsilon'],
  caption: 'Land cover in; NDWI out. Water is positive, land negative.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const green = uploadBand(builder, imagery, 'green', imagery.green);
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const shortwave = uploadBand(builder, imagery, 'swir', imagery.shortwaveInfrared);
    const output = builder.allocate('ndwi', 'float32', cellCount);
    const outputValidity = builder.allocate('ndwi-valid', 'uint32', cellCount);
    const common = {
      width: imagery.columns,
      height: imagery.rows,
      epsilon: values.epsilon,
      nearInfrared: nearInfrared.band,
      output: output.view,
      outputValidity: outputValidity.view
    };
    const index =
      values.method === 0
        ? new GPURasterNDWI({...common, method: 'mcfeeters', green: green.band})
        : new GPURasterNDWI({...common, method: 'gao', shortwaveInfrared: shortwave.band});
    index.addToGraph(builder.graph);
    return {outputs: {green: green.buffer, output}};
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.green, {context: true}),
    drawRaster(inputs, outputs.output, {ramp: 'diverging', range: [-1, 1], smoothing: true})
  ]
});
