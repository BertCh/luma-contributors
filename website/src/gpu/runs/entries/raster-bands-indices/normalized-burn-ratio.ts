// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterNBR on Sentinel-2 imagery: vegetation (high NIR, low SWIR2) is strongly positive, bare
// soil sits near zero. The extract is pre-fire, so a post-fire scenario is derived from the
// measured NDVI drop between July and September 2021 (burned cells lose NIR and gain SWIR2); the
// pointer swipes pre-fire NBR against post-fire NBR. `epsilon` and `clamp` are construction props.

import {GPURasterNBR} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './normalized-burn-ratio.parameters';
import {drawRaster, uploadBand} from './support/bands';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';
import {createPostFireBands} from './support/indices';

export default defineOperationRun({
  parameters,
  rebuildOn: ['epsilon', 'clampMaximum'],
  caption:
    'NIR and SWIR2 in; pre-fire NBR left, scenario post-fire NBR right (burned cells from the measured NDVI drop).',
  createInputs: ({world}) => {
    const imagery = createImageryBands(world);
    return {
      ...imagery,
      post: createPostFireBands(
        imagery.columns,
        imagery.rows,
        imagery.nearInfrared,
        imagery.shortwaveInfrared2
      )
    };
  },
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const before = {
      nearInfrared: uploadBand(builder, imagery, 'nir', imagery.nearInfrared),
      shortwave: uploadBand(builder, imagery, 'swir2', imagery.shortwaveInfrared2)
    };
    const after = {
      nearInfrared: uploadBand(builder, imagery, 'nir-post', imagery.post.nearInfrared),
      shortwave: uploadBand(builder, imagery, 'swir2-post', imagery.post.shortwaveInfrared2)
    };
    const common = {
      width: imagery.columns,
      height: imagery.rows,
      epsilon: values.epsilon,
      clamp: [-values.clampMaximum, values.clampMaximum] as [number, number]
    };
    const output = builder.allocate('nbr', 'float32', cellCount);
    const postOutput = builder.allocate('nbr-post', 'float32', cellCount);
    builder.graph.add(
      new GPURasterNBR({
        ...common,
        nearInfrared: before.nearInfrared.band,
        shortwaveInfrared2: before.shortwave.band,
        output: output.view,
        outputValidity: builder.allocate('nbr-valid', 'uint32', cellCount).view
      })
    );
    builder.graph.add(
      new GPURasterNBR({
        ...common,
        id: 'nbr-post',
        nearInfrared: after.nearInfrared.band,
        shortwaveInfrared2: after.shortwave.band,
        output: postOutput.view,
        outputValidity: builder.allocate('nbr-post-valid', 'uint32', cellCount).view
      })
    );
    return {outputs: {output, postOutput}};
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe pre and post fire',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const {index} = getCellAt(inputs, world, pointer);
      const nbr = (nir: number, swir: number) => (nir - swir) / (nir + swir + values.epsilon);
      const before = nbr(inputs.nearInfrared[index], inputs.shortwaveInfrared2[index]);
      const after = nbr(inputs.post.nearInfrared[index], inputs.post.shortwaveInfrared2[index]);
      const readout = `NBR ${formatNumber(before)} to ${formatNumber(after)}`;
      return getSwipeResult(pointer, readout);
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    return [
      drawRaster(inputs, outputs.output, {
        ramp: 'diverging',
        range: [-1, 1],
        smoothing: true,
        clip: {side: 'left', x: curtain}
      }),
      drawRaster(inputs, outputs.postOutput, {
        ramp: 'diverging',
        range: [-1, 1],
        smoothing: true,
        clip: {side: 'right', x: curtain}
      })
    ];
  }
});
