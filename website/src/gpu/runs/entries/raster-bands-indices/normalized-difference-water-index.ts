// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterNDWI on Sentinel-2 imagery: water reads high. Both recipes run side by side, McFeeters
// (green, NIR) and Gao (NIR, SWIR), and the pointer swipes between them. The extract is the
// Feather River watershed before the fire and has no open water, so both indices are negative;
// the fixed range shows the vegetation-to-soil spread instead of a uniform tint.

import {GPURasterNDWI} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './normalized-difference-water-index.parameters';
import {drawRaster, uploadBand} from './support/bands';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';

export default defineOperationRun({
  parameters,
  rebuildOn: ['epsilon'],
  caption:
    'Land cover in; McFeeters NDWI (green, NIR) left, Gao NDWI (NIR, SWIR) right. No open water in this extract.',
  createInputs: ({world}) => createImageryBands(world),
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const green = uploadBand(builder, imagery, 'green', imagery.green);
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const shortwave = uploadBand(builder, imagery, 'swir', imagery.shortwaveInfrared);
    const mcfeeters = builder.allocate('ndwi-mcfeeters', 'float32', cellCount);
    const gao = builder.allocate('ndwi-gao', 'float32', cellCount);
    const common = {
      width: imagery.columns,
      height: imagery.rows,
      epsilon: values.epsilon,
      nearInfrared: nearInfrared.band
    };
    builder.graph.add(
      new GPURasterNDWI({
        ...common,
        method: 'mcfeeters',
        green: green.band,
        output: mcfeeters.view,
        outputValidity: builder.allocate('mcfeeters-valid', 'uint32', cellCount).view
      })
    );
    builder.graph.add(
      new GPURasterNDWI({
        ...common,
        id: 'ndwi-gao',
        method: 'gao',
        shortwaveInfrared: shortwave.band,
        output: gao.view,
        outputValidity: builder.allocate('gao-valid', 'uint32', cellCount).view
      })
    );
    return {outputs: {mcfeeters, gao}};
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe McFeeters and Gao',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const {index} = getCellAt(inputs, world, pointer);
      const green = inputs.green[index];
      const nir = inputs.nearInfrared[index];
      const swir = inputs.shortwaveInfrared[index];
      const readout = `McFeeters ${formatNumber((green - nir) / (green + nir + values.epsilon))} | Gao ${formatNumber((nir - swir) / (nir + swir + values.epsilon))}`;
      return getSwipeResult(pointer, readout);
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    const look = {ramp: 'diverging', range: [-0.8, 0.8], smoothing: true} as const;
    return [
      drawRaster(inputs, outputs.mcfeeters, {...look, clip: {side: 'left', x: curtain}}),
      drawRaster(inputs, outputs.gao, {...look, clip: {side: 'right', x: curtain}})
    ];
  }
});
