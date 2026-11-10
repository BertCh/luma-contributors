// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEVI on Sentinel-2 imagery. The blue band carries atmospheric haze; EVI subtracts it
// (the 7.5 blue term), so raising the haze slider changes the index. NDVI of the same bands is
// computed alongside by GPURasterArithmetic, and the pointer swipes between the two.

import {GPURasterEVI} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './enhanced-vegetation-index.parameters';
import {drawRaster, uploadBand} from './support/bands';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';
import {addNormalizedDifference} from './support/indices';

const RANGE = [-0.2, 1] as const;

export default defineOperationRun({
  parameters,
  caption: 'NIR, red and blue in; NDVI on the left, the haze-corrected EVI on the right.',
  createInputs: ({world}) => createImageryBands(world),
  build(builder, imagery) {
    const cellCount = imagery.values.length;
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const red = uploadBand(builder, imagery, 'red', imagery.red);
    const blue = uploadBand(builder, imagery, 'blue', imagery.blue);
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
    const ndvi = addNormalizedDifference(builder, 'ndvi', nearInfrared.buffer, red.buffer);
    return {
      outputs: {ndvi, output},
      update: current => blue.buffer.write(imagery.blue.map(value => value + current.haze))
    };
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe NDVI and EVI',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const {index} = getCellAt(inputs, world, pointer);
      const nir = inputs.nearInfrared[index];
      const red = inputs.red[index];
      const blue = inputs.blue[index] + values.haze;
      const ndvi = (nir - red) / (nir + red);
      const evi = (2.5 * (nir - red)) / (nir + 6 * red - 7.5 * blue + 1);
      const readout = `NDVI ${formatNumber(ndvi)} | EVI ${formatNumber(evi)}`;
      return getSwipeResult(pointer, readout);
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    return [
      drawRaster(inputs, outputs.ndvi, {
        ramp: 'sequential',
        range: RANGE,
        smoothing: true,
        clip: {side: 'left', x: curtain}
      }),
      drawRaster(inputs, outputs.output, {
        ramp: 'sequential',
        range: RANGE,
        smoothing: true,
        clip: {side: 'right', x: curtain}
      })
    ];
  }
});
