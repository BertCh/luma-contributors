// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainDEMCoregistration between a reference DEM and a second epoch that is the same terrain
// plus a constant vertical bias, plus real change (a thinning glacier) on unstable ground. The
// median shift is fitted on the stable mask only, and the corrected difference is drawn: the bias
// is gone, the glacier loss remains. (The output interleaves difference and standard deviation, so
// each cell is two pixels wide: difference left, uncertainty right.)

import {GPUTerrainDEMCoregistration} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-dem-coregistration.parameters';
import {createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const NOISE_STANDARD_DEVIATION = 0.002;

export default defineOperationRun({
  parameters,
  caption:
    'Corrected difference between two epochs: the constant bias is removed, the glacier loss stays.',
  createInputs: context => {
    const terrain = createTerrainInputs(context, 36);
    const unstable = new Uint32Array(terrain.cellCount);
    const glacierChange = new Float32Array(terrain.cellCount);
    const noise = new Float32Array(terrain.cellCount);
    for (let cell = 0; cell < terrain.cellCount; cell++) {
      const x = ((cell % terrain.columns) + 0.5) / terrain.columns;
      const y = (Math.floor(cell / terrain.columns) + 0.5) / terrain.rows;
      const distance = Math.hypot((x - 0.3) * (terrain.columns / terrain.rows), y - 0.45);
      const glacier = Math.exp(-((distance / 0.22) ** 2));
      glacierChange[cell] = -glacier;
      unstable[cell] = glacier > 0.1 ? 1 : 0;
      noise[cell] = (context.random() - 0.5) * 2 * NOISE_STANDARD_DEVIATION;
    }
    return {...terrain, stableMask: unstable.map(value => 1 - value), glacierChange, noise};
  },
  build(builder, terrain) {
    const reference = builder.upload('reference', 'float32', terrain.elevation);
    const moving = builder.allocate('moving', 'float32', terrain.cellCount);
    const stableMask = builder.upload('stable-mask', 'uint32', terrain.stableMask);
    const differenceVariance = builder.upload(
      'difference-variance',
      'float32',
      new Float32Array(terrain.cellCount).fill(NOISE_STANDARD_DEVIATION ** 2)
    );
    const differences = builder.allocate('differences', 'float32', terrain.cellCount * 2);
    builder.graph.add(
      new GPUTerrainDEMCoregistration({
        grid: terrain.grid,
        reference: reference.view,
        moving: moving.view,
        stableMask: stableMask.view,
        differenceVariance: differenceVariance.view,
        output: {
          fit: builder.allocate('fit', 'float32', 4).view,
          differences: differences.view,
          validity: builder.allocate('validity', 'uint32', terrain.cellCount).view,
          volume: builder.allocate('volume', 'float32', 3).view
        }
      })
    );
    return {
      outputs: {reference, differences},
      update: current => {
        const epoch = new Float32Array(terrain.cellCount);
        for (let cell = 0; cell < terrain.cellCount; cell++) {
          epoch[cell] =
            terrain.elevation[cell] +
            current.bias +
            current.thinning * terrain.glacierChange[cell] +
            terrain.noise[cell];
        }
        moving.write(epoch);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.reference),
    {
      layer: 'raster',
      values: outputs.differences,
      columns: inputs.columns * 2,
      rows: inputs.rows,
      ramp: 'diverging',
      range: [-0.1, 0.1],
      alpha: 0.95
    }
  ]
});
