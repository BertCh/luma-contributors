// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// makeGPURasterBand wraps raw digital numbers with their calibration. Here green and near-infrared
// Landsat-style integers (reflectance = DN * scale + offset, fill value 0 under a cloud) feed
// GPURasterNDWI. The band carries the calibration, so the index is computed on reflectance;
// wrong `scale` or `offset` shifts the index, and without `noDataValue` the cloud's fill value
// is read as a real (negative) reflectance. Band metadata is a construction prop, so the run
// rebuilds on every change.

import {
  GPURasterNDWI,
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands, createSmoothField} from '../../inputs';
import {parameters} from './make-raster-band.parameters';
import {drawRaster} from './support/bands';

const TRUE_SCALE = 0.0000275;
const TRUE_OFFSET = -0.2;

export default defineOperationRun({
  parameters,
  rebuildOn: ['scale', 'offset', 'noDataValue'],
  caption: 'Raw integers and their calibration in; the index on calibrated reflectance out.',
  createInputs: ({random, world}) => {
    const imagery = createImageryBands(random, world, 56);
    const cloud = createSmoothField(random, 3, 3);
    const toDigital = (reflectance: Float32Array) =>
      Float32Array.from(reflectance, (value, index) =>
        cloud(
          ((index % imagery.columns) + 0.5) / imagery.columns,
          (Math.floor(index / imagery.columns) + 0.5) / imagery.rows
        ) > 0.78
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
    const output = builder.allocate('ndwi', 'float32', cellCount);
    const outputValidity = builder.allocate('ndwi-valid', 'uint32', cellCount);
    new GPURasterNDWI({
      method: 'mcfeeters',
      width: inputs.columns,
      height: inputs.rows,
      green: makeGPURasterBand(green.view, grid, {id: 'green', ...calibration}),
      nearInfrared: makeGPURasterBand(nearInfrared.view, grid, {id: 'nir', ...calibration}),
      output: output.view,
      outputValidity: outputValidity.view
    }).addToGraph(builder.graph);
    return {outputs: {green, output}};
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.green, {context: true}),
    drawRaster(inputs, outputs.output, {ramp: 'diverging', range: [-1, 1], smoothing: true})
  ]
});
