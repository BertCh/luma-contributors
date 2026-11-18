// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// makeGPURasterBand wraps raw digital numbers with their calibration. Here green and near-infrared
// Landsat-style integers (reflectance = DN * scale + offset, fill value 0 under a cloud) feed
// GPURasterNDWI twice: left of the curtain the integers are read raw, as if they were
// reflectance (no scale, offset or noDataValue); right of it the band carries the sliders'
// calibration, so the index is computed on reflectance and the cloud is nodata. Band metadata is
// a construction prop, so the run rebuilds on every slider change. The extract has no open water,
// so the index shows the vegetation-to-soil spread rather than a lake.

import {
  GPURasterNDWI,
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands, createSmoothField} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './make-raster-band.parameters';
import {drawRaster} from './support/bands';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';

const TRUE_SCALE = 0.0000275;
const TRUE_OFFSET = -0.2;

export default defineOperationRun({
  parameters,
  rebuildOn: ['scale', 'offset', 'noDataValue'],
  caption:
    'Raw integers in; NDWI read raw (left) and on calibrated reflectance with the cloud as nodata (right).',
  createInputs: ({random, world}) => {
    const imagery = createImageryBands(world, 56);
    const cloud = createSmoothField(random, 3, 3);
    const toDigital = (reflectance: Float32Array) =>
      Float32Array.from(reflectance, (value, index) =>
        cloud(
          ((index % imagery.columns) + 0.5) / imagery.columns,
          (Math.floor(index / imagery.columns) + 0.5) / imagery.rows
        ) > 0.66
          ? 0
          : Math.round((value - TRUE_OFFSET) / TRUE_SCALE)
      );
    return {
      ...imagery,
      greenDigital: toDigital(imagery.green),
      nearInfraredDigital: toDigital(imagery.nearInfrared)
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const grid = makeGPURasterMetadata({width: inputs.columns, height: inputs.rows});
    const green = builder.upload('green', 'float32', inputs.greenDigital);
    const nearInfrared = builder.upload('nir', 'float32', inputs.nearInfraredDigital);
    const calibration = {
      scale: values.scale * 1e-5,
      offset: values.offset,
      ...(values.noDataValue === 1 ? {noDataValue: 0} : {})
    };
    const addIndex = (
      id: string,
      settings: {scale?: number; offset?: number; noDataValue?: number}
    ) => {
      const output = builder.allocate(`ndwi-${id}`, 'float32', cellCount);
      builder.graph.add(
        new GPURasterNDWI({
          id: `ndwi-${id}`,
          method: 'mcfeeters',
          width: inputs.columns,
          height: inputs.rows,
          green: makeGPURasterBand(green.view, grid, {id: `green-${id}`, ...settings}),
          nearInfrared: makeGPURasterBand(nearInfrared.view, grid, {id: `nir-${id}`, ...settings}),
          output: output.view,
          outputValidity: builder.allocate(`ndwi-${id}-valid`, 'uint32', cellCount).view
        })
      );
      return output;
    };
    return {outputs: {raw: addIndex('raw', {}), output: addIndex('calibrated', calibration)}};
  },
  interaction: {
    kind: 'swipe',
    hint: 'swipe raw and calibrated',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const {index} = getCellAt(inputs, world, pointer);
      const digital = inputs.greenDigital[index];
      const reflectance = digital * values.scale * 1e-5 + values.offset;
      const readout =
        digital === 0 && values.noDataValue === 1
          ? 'DN 0 | nodata (cloud)'
          : `DN ${digital} | reflectance ${formatNumber(reflectance, 3)}`;
      return getSwipeResult(pointer, readout);
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    const look = {ramp: 'sequential', range: [-0.9, 0.1], smoothing: true} as const;
    return [
      drawRaster(inputs, outputs.raw, {...look, clip: {side: 'left', x: curtain}}),
      drawRaster(inputs, outputs.output, {...look, clip: {side: 'right', x: curtain}})
    ];
  }
});
