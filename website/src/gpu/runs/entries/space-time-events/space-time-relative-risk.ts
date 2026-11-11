// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeRelativeRisk of GOES lightning flashes over the Midwest: cases are the brightest 30%
// of flashes by optical energy, controls the other 70%, both taken at the same time slice. The
// raster is log(case density / control density): red where bright flashes are over-represented at
// that place and hour. Cells with little control density are left empty. The pointer is the clock:
// across is the slice time, up is the temporal bandwidth.

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  getGPUSpaceTimeKernelDensityParameterValues,
  GPUSpaceTimeRelativeRisk,
  GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createLightningFlashes} from './_lib/events';
import {addKernelPass} from './_lib/kernel';
import {createTimeWindowGlyphs, getTimeWeights, OBSERVATION_HOURS} from './_lib/time-window';
import {parameters} from './space-time-relative-risk.parameters';

const FLASH_COUNT = 3000;
const ROWS = 54;
const CASE_SHARE = 0.3;
/** Control probability density below which a cell is left empty (the mean is about 0.03). */
const MINIMUM_CONTROL_DENSITY = 0.004;

type Inputs = ReturnType<typeof createLightningFlashes> & {
  casePositions: Float32Array;
  caseHours: Float32Array;
  controlPositions: Float32Array;
  controlHours: Float32Array;
  width: number;
  columns: number;
  grid: ReturnType<typeof makeGPURasterMetadata>;
};
type Outputs = {casePositions: RunBuffer; recency: RunBuffer; risk: RunBuffer};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'GOES lightning, May 21–22 2024 (3,000 of 4,096 sampled flashes): log relative risk of the brightest 30% of flashes against the rest, at the slice time.',
  createInputs: ({world}) => {
    const columns = Math.round(ROWS * world.width);
    const flashes = createLightningFlashes([0, 0, world.width, world.height], FLASH_COUNT);
    const threshold = Float32Array.from(flashes.energy).sort()[
      Math.floor(FLASH_COUNT * (1 - CASE_SHARE))
    ];
    const isCase = Array.from(flashes.energy, energy => energy >= threshold);
    const select = (rows: boolean, data: Float32Array, size: number) =>
      Float32Array.from(
        isCase.flatMap((flag, row) =>
          flag === rows ? Array.from(data.subarray(row * size, row * size + size)) : []
        )
      );
    return {
      ...flashes,
      casePositions: select(true, flashes.positions, 2),
      caseHours: select(true, flashes.hours, 1),
      controlPositions: select(false, flashes.positions, 2),
      controlHours: select(false, flashes.hours, 1),
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
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH
    );
    const cellCount = flashes.columns * ROWS;
    const logRisk = builder.allocate('log-risk', 'float32', cellCount);
    const controlDensity = builder.allocate('control-density', 'float32', cellCount);
    const casePositions = builder.upload('case-positions', 'float32x2', flashes.casePositions);
    builder.graph.add(
      new GPUSpaceTimeRelativeRisk({
        casePositions: casePositions.view,
        caseTimes: builder.upload('case-times', 'float32', flashes.caseHours).view,
        controlPositions: builder.upload('control-positions', 'float32x2', flashes.controlPositions)
          .view,
        controlTimes: builder.upload('control-times', 'float32', flashes.controlHours).view,
        parameters: parameterBuffer.view,
        grid: flashes.grid,
        sliceCount: 1,
        indexGridSize: [32, 20],
        indexBounds: [0, 0, flashes.width, 1],
        output: {values: logRisk.view, controlDensity: controlDensity.view}
      })
    );
    // Leave cells with almost no controls empty: there the ratio is two tails divided.
    const risk = builder.allocate('risk', 'float32', cellCount);
    addKernelPass(builder, {
      id: 'relative-risk-mask',
      invocationCount: cellCount,
      bindings: [
        {name: 'logRisk', buffer: logRisk, access: 'read'},
        {name: 'controlDensity', buffer: controlDensity, access: 'read'},
        {name: 'risk', buffer: risk, access: 'read_write'}
      ],
      body: `var notANumberBits = 0x7fc00000u;
  let control = controlDensity[controlDensityOffset + index];
  risk[riskOffset + index] = select(bitcast<f32>(notANumberBits), logRisk[logRiskOffset + index], control >= ${MINIMUM_CONTROL_DENSITY});`
    });
    const recency = builder.allocate('recency', 'float32', flashes.caseHours.length);
    return {
      outputs: {casePositions, recency, risk},
      update: current => {
        recency.write(
          getTimeWeights(flashes.caseHours, current.firstSliceTime, current.temporalBandwidth)
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
        inputs.caseHours,
        values.firstSliceTime,
        values.temporalBandwidth,
        world.width,
        'case flashes'
      )
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.risk,
      columns: inputs.columns,
      rows: ROWS,
      ramp: 'diverging',
      range: [-2, 2],
      smoothing: true
    },
    {
      layer: 'dots',
      positions: outputs.casePositions,
      values: outputs.recency,
      range: [0, 1],
      ramp: 'grey',
      fade: true,
      radius: 2
    }
  ]
});
