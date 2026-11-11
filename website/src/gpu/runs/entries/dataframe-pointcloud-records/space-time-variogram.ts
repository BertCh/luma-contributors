// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeVariogram of a field that also changes over time. The grid is the semivariance per
// (spatial lag, temporal lag) bin: columns are time lags, rows are distance bins from near (top)
// to far. It grows along both axes when the field varies in space and in time.

import {
  getGPUSpaceTimeVariogramParameterValues,
  GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH,
  GPUSpaceTimeVariogram
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {getRasterCell} from '../../pick';
import {sampleFlashes} from '../../../data/lightning';
import {getVariance} from './_lib/helpers';
import {parameters} from './space-time-variogram.parameters';

const POINT_COUNT = 900;
const LAG_COUNT = 16;
const TEMPORAL_LAG_COUNT = 8;
// Class 0 of each axis is the exact zero lag, so the output is (L + 1) x (T + 1) bins.
const ROWS = LAG_COUNT + 1;
const COLUMNS = TEMPORAL_LAG_COUNT + 1;
// Tick segments, from the loops below: one per second time lag (lags 0, 2, ...) and one per fourth
// distance bin (bins 4, 8, ...).
const TIME_TICK_STEP = 2;
const DISTANCE_TICK_STEP = 4;
const TICK_SEGMENT_COUNT =
  Math.ceil(COLUMNS / TIME_TICK_STEP) + Math.floor((ROWS - 1) / DISTANCE_TICK_STEP);

const INSET = [0.03, 0.04, 0.24, 0.3] as const;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {semivariances: RunBuffer; ticks: RunBuffer; inset: RunBuffer};

function createInputs({world}: {world: {width: number; height: number}}) {
  const flashes = sampleFlashes(world, POINT_COUNT);
  const field = Float32Array.from(flashes.energy, Math.log1p);
  return {...flashes, world, field, variance: getVariance(field)};
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Semivariance per distance (rows, near at top) and time lag (columns), on a range of twice the field variance (the first row and column are the exact zero lags); the inset is the flash map.',
  createInputs,
  build(builder, inputs) {
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const values = builder.upload('values', 'float32', inputs.field);
    const times = builder.upload('times', 'float32', inputs.hours);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH
    );
    const semivariances = builder.allocate('semivariances', 'float32', ROWS * COLUMNS);
    builder.graph.add(
      new GPUSpaceTimeVariogram({
        positions: positions.view,
        values: values.view,
        times: times.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        lagCount: LAG_COUNT,
        temporalLagCount: TEMPORAL_LAG_COUNT,
        output: {
          semivariances: semivariances.view,
          pairCounts: builder.allocate('pair-counts', 'uint32', ROWS * COLUMNS).view,
          status: builder.allocate('status', 'uint32', ROWS * COLUMNS).view
        }
      })
    );
    // Tick marks: every second time lag along the top, every fourth distance bin on the left.
    const ticks: number[] = [];
    for (let lag = 0; lag < COLUMNS; lag += TIME_TICK_STEP) {
      const x = (lag / COLUMNS) * inputs.world.width;
      ticks.push(x, 1, x, 0.96);
    }
    for (let bin = DISTANCE_TICK_STEP; bin < ROWS; bin += DISTANCE_TICK_STEP) {
      ticks.push(0, 1 - bin / ROWS, 0.015, 1 - bin / ROWS);
    }
    const inset = builder.upload(
      'inset',
      'float32x2',
      Float32Array.from({length: POINT_COUNT * 2}, (_, index) => {
        const row = index >> 1;
        return index % 2 === 0
          ? (INSET[0] + (inputs.positions[row * 2] / inputs.world.width) * (INSET[2] - INSET[0])) *
              inputs.world.width
          : INSET[1] + inputs.positions[row * 2 + 1] * (INSET[3] - INSET[1]);
      })
    );
    return {
      outputs: {
        semivariances,
        inset,
        ticks: builder.upload('ticks', 'float32x4', Float32Array.from(ticks))
      },
      update: current => {
        parameterBuffer.write(
          getGPUSpaceTimeVariogramParameterValues({
            // The flashes live in the drawing world, [0, width] x [0, 1].
            bounds: [0, 0, inputs.world.width, 1],
            maximumDistance: current.maximumDistance,
            maximumTimeLag: current.maximumTimeLag
          })
        );
      }
    };
  },
  rebuildOn: [],
  // The pointer is a pad on the two extents: x is the largest time lag, y the largest distance.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'time lag across, distance up',
    x: 'maximumTimeLag',
    y: 'maximumDistance',
    decorate: (values, pointer, {inputs, outputs, peek}) => {
      const cell = getRasterCell(
        COLUMNS,
        ROWS,
        [0, 0, inputs.world.width, 1],
        pointer.x,
        pointer.y
      );
      const gamma = peek(outputs.semivariances)?.[cell.index];
      const left = (cell.column / COLUMNS) * inputs.world.width;
      const right = ((cell.column + 1) / COLUMNS) * inputs.world.width;
      return {
        glyphs: [
          {
            shape: 'box',
            minimum: [left, 1 - (cell.row + 1) / ROWS],
            maximum: [right, 1 - cell.row / ROWS]
          }
        ],
        readout: `distance ≤ ${values.maximumDistance.toFixed(2)} · lag ≤ ${values.maximumTimeLag} h · γ ${
          gamma === undefined || !Number.isFinite(gamma) ? '…' : gamma.toFixed(1)
        }`
      };
    }
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.semivariances,
      columns: COLUMNS,
      rows: ROWS,
      range: [0, 2 * inputs.variance],
      bands: 10,
      smoothing: true
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.ticks,
      count: TICK_SEGMENT_COUNT,
      color: [1, 1, 1, 0.8]
    },
    {layer: 'dots', context: true, positions: outputs.inset, color: [1, 1, 1, 0.7], radius: 1.6}
  ]
});
