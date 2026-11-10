// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeVariogram of a field that also changes over time. The grid is the semivariance per
// (spatial lag, temporal lag) bin: columns are time lags, rows are distance bins from near (top)
// to far. It grows along both axes when the field varies in space and in time.

import {GPUSpaceTimeVariogram} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {getRasterCell} from '../../pick';
import {sampleFlashes} from '../../../data/lightning';
import {getVariance} from './_lib/helpers';
import {parameters} from './space-time-variogram.parameters';

const POINT_COUNT = 900;
const LAG_COUNT = 16;
const TEMPORAL_LAG_COUNT = 8;

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
    'Semivariance per distance (rows, near at top) and time lag (columns), on a range of twice the field variance; the inset is the flash map.',
  createInputs,
  build(builder, inputs) {
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const values = builder.upload('values', 'float32', inputs.field);
    const times = builder.upload('times', 'float32', inputs.hours);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const semivariances = builder.allocate(
      'semivariances',
      'float32',
      LAG_COUNT * TEMPORAL_LAG_COUNT
    );
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
          pairCounts: builder.allocate('pair-counts', 'uint32', LAG_COUNT * TEMPORAL_LAG_COUNT)
            .view,
          status: builder.allocate('status', 'uint32', LAG_COUNT * TEMPORAL_LAG_COUNT).view
        }
      })
    );
    // Tick marks: every second time lag along the top, every fourth distance bin on the left.
    const ticks: number[] = [];
    for (let lag = 0; lag < TEMPORAL_LAG_COUNT; lag += 2) {
      const x = (lag / TEMPORAL_LAG_COUNT) * inputs.world.width;
      ticks.push(x, 1, x, 0.96);
    }
    for (let bin = 4; bin < LAG_COUNT; bin += 4) {
      ticks.push(0, 1 - bin / LAG_COUNT, 0.015, 1 - bin / LAG_COUNT);
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
        parameterBuffer.write(Float32Array.of(current.maximumDistance, current.maximumTimeLag));
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
        TEMPORAL_LAG_COUNT,
        LAG_COUNT,
        [0, 0, inputs.world.width, 1],
        pointer.x,
        pointer.y
      );
      const gamma = peek(outputs.semivariances)?.[cell.index];
      const left = (cell.column / TEMPORAL_LAG_COUNT) * inputs.world.width;
      const right = ((cell.column + 1) / TEMPORAL_LAG_COUNT) * inputs.world.width;
      return {
        glyphs: [
          {
            shape: 'box',
            minimum: [left, 1 - (cell.row + 1) / LAG_COUNT],
            maximum: [right, 1 - cell.row / LAG_COUNT]
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
      columns: TEMPORAL_LAG_COUNT,
      rows: LAG_COUNT,
      range: [0, 2 * inputs.variance],
      bands: 10,
      smoothing: true
    },
    {layer: 'lines', context: true, segments: outputs.ticks, count: 7, color: [1, 1, 1, 0.8]},
    {layer: 'dots', context: true, positions: outputs.inset, color: [1, 1, 1, 0.7], radius: 1.6}
  ]
});
