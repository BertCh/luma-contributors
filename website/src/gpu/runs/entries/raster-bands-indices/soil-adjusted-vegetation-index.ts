// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSAVI on Sentinel-2 imagery. `soilAdjustment` (L) is a construction prop: zero gives
// NDVI, larger values flatten the soil response. A second SAVI with L = 0 is the NDVI baseline the
// pointer swipes against. The run rebuilds when the slider moves.

import {GPURasterSAVI} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './soil-adjusted-vegetation-index.parameters';
import {drawRaster, uploadBand} from './support/bands';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';

export default defineOperationRun({
  parameters,
  rebuildOn: ['soilAdjustment', 'epsilon'],
  caption: 'NIR and red in; NDVI (L 0) on the left, SAVI with the soil correction L on the right.',
  createInputs: ({world}) => createImageryBands(world),
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const nearInfrared = uploadBand(builder, imagery, 'nir', imagery.nearInfrared);
    const red = uploadBand(builder, imagery, 'red', imagery.red);
    const output = builder.allocate('savi', 'float32', cellCount);
    const outputValidity = builder.allocate('savi-valid', 'uint32', cellCount);
    const baseline = builder.allocate('ndvi', 'float32', cellCount);
    const baselineValidity = builder.allocate('ndvi-valid', 'uint32', cellCount);
    const common = {
      width: imagery.columns,
      height: imagery.rows,
      nearInfrared: nearInfrared.band,
      red: red.band,
      epsilon: values.epsilon
    };
    builder.graph.add(
      new GPURasterSAVI({
        ...common,
        soilAdjustment: values.soilAdjustment,
        output: output.view,
        outputValidity: outputValidity.view
      })
    );
    builder.graph.add(
      new GPURasterSAVI({
        ...common,
        id: 'ndvi-baseline',
        soilAdjustment: 0,
        output: baseline.view,
        outputValidity: baselineValidity.view
      })
    );
    return {outputs: {baseline, output}};
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe NDVI and SAVI',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const {index} = getCellAt(inputs, world, pointer);
      const nir = inputs.nearInfrared[index];
      const red = inputs.red[index];
      const adjustment = values.soilAdjustment;
      const savi = ((nir - red) / (nir + red + adjustment + values.epsilon)) * (1 + adjustment);
      const readout = `L 0: ${formatNumber((nir - red) / (nir + red))} | L ${adjustment}: ${formatNumber(savi)}`;
      return getSwipeResult(pointer, readout);
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    return [
      drawRaster(inputs, outputs.baseline, {
        ramp: 'sequential',
        range: [0, 1],
        smoothing: true,
        clip: {side: 'left', x: curtain}
      }),
      drawRaster(inputs, outputs.output, {
        ramp: 'sequential',
        range: [0, 1],
        smoothing: true,
        clip: {side: 'right', x: curtain}
      })
    ];
  }
});
