// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeKernelDensity of GOES lightning flashes over the Midwest: one time slice of the
// space-time intensity at `firstSliceTime`, with a spatial and a temporal Gaussian kernel. Flashes
// are drawn brighter the closer their time is to the slice. The pointer is the clock: across is the
// slice time, up is the temporal bandwidth; the bar along the bottom is the time window.

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  getGPUSpaceTimeKernelDensityParameterValues,
  GPUSpaceTimeKernelDensity,
  GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createLightningFlashes} from './_lib/events';
import {createTimeWindowGlyphs, getTimeWeights, OBSERVATION_HOURS} from './_lib/time-window';
import {parameters} from './space-time-kernel-density.parameters';

const FLASH_COUNT = 3000;
const ROWS = 54;

type Inputs = ReturnType<typeof createLightningFlashes> & {
  width: number;
  columns: number;
  grid: ReturnType<typeof makeGPURasterMetadata>;
};
type Outputs = {positions: RunBuffer; recency: RunBuffer; density: RunBuffer};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'GOES lightning, May 21–22 2024 (3,000 of 4,096 sampled flashes): space-time kernel intensity at the slice time; flashes brighten near that time.',
  createInputs: ({world}) => {
    const columns = Math.round(ROWS * world.width);
    return {
      ...createLightningFlashes([0, 0, world.width, world.height], FLASH_COUNT),
      width: world.width,
      columns,
      grid: makeGPURasterMetadata({
        width: columns,
        height: ROWS,
        cellSize: [world.width / columns, 1 / ROWS],
        origin: [0, 1]
      })
    };
  },
  build(builder, flashes) {
    const positions = builder.upload('positions', 'float32x2', flashes.positions);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH
    );
    const density = builder.allocate('density', 'float32', flashes.columns * ROWS);
    builder.graph.add(
      new GPUSpaceTimeKernelDensity({
        positions: positions.view,
        times: builder.upload('times', 'float32', flashes.hours).view,
        parameters: parameterBuffer.view,
        grid: flashes.grid,
        sliceCount: 1,
        indexGridSize: [32, 20],
        indexBounds: [0, 0, flashes.width, 1],
        output: {values: density.view}
      })
    );
    const recency = builder.allocate('recency', 'float32', flashes.count);
    return {
      outputs: {positions, recency, density},
      update: current => {
        recency.write(
          getTimeWeights(flashes.hours, current.firstSliceTime, current.temporalBandwidth)
        );
        parameterBuffer.write(
          getGPUSpaceTimeKernelDensityParameterValues(
            {
              bandwidth: current.bandwidth,
              temporalBandwidth: current.temporalBandwidth,
              firstSliceTime: current.firstSliceTime,
              sliceInterval: 1,
              timeWindow: [0, OBSERVATION_HOURS]
            },
            flashes.grid,
            1
          )
        );
      }
    };
  },
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'time',
    hint: 'time across, window up',
    x: 'firstSliceTime',
    y: 'temporalBandwidth',
    decorate: (values, _pointer, {inputs, world}) =>
      createTimeWindowGlyphs(
        inputs.hours,
        values.firstSliceTime,
        values.temporalBandwidth,
        world.width
      )
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.density,
      columns: inputs.columns,
      rows: ROWS,
      range: [6, 13],
      logarithmic: true,
      smoothing: true
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.recency,
      range: [0, 1],
      ramp: 'grey',
      fade: true,
      radius: 2
    }
  ]
});
