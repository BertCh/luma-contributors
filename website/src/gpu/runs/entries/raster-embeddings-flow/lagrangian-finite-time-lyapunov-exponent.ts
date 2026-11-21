// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFiniteTimeLyapunovExponent: a dense grid of seeds is carried through the Helene wind by
// GPULagrangianFlowMap, and the FTLE of each seed measures how fast its neighbours separate over
// `integrationTime`. Ridges are the lines that split the flow (forward) or gather it (backward).
// The pointer's x is the signed integration time, zero at the centre: short times show local shear,
// long ones sharp ridges, on a fixed range so the sharpening is visible.
// Steady field (one snapshot).

import {
  getGPUFiniteTimeLyapunovExponentParameterValues,
  getGPULagrangianFlowMapParameterValues,
  GPUFiniteTimeLyapunovExponent,
  GPULagrangianFlowMap
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes} from '../../interaction';
import {addRowFlip} from './_lib/kernels';
import {createWind} from './_lib/wind';
import {parameters} from './lagrangian-finite-time-lyapunov-exponent.parameters';

const SEED_ROWS = 64;
const STEP_COUNT = 64;
const MIN_TIME = 0.1;
/** Fixed FTLE range (already divided by |T|) so ridges visibly sharpen as |T| grows. */
const FTLE_RANGE = [0, 1.2] as const;

export default defineOperationRun({
  parameters,
  caption:
    'FTLE of the Helene wind: left of centre integrates backward (attracting ridges), right forward (repelling). Steady wind; time units are uncalibrated (1 m/s = 0.01 field heights per unit).',
  createInputs: ({aspect}) => {
    const field = createWind(aspect);
    const width = field.columns * field.cellSize;
    const seedColumns = Math.round(SEED_ROWS * width);
    const seeds = Float32Array.from({length: seedColumns * SEED_ROWS * 2}, (_, item) => {
      const seed = item >> 1;
      return item & 1
        ? (Math.floor(seed / seedColumns) + 0.5) / SEED_ROWS
        : (((seed % seedColumns) + 0.5) * width) / seedColumns;
    });
    return {...field, width, seeds, seedColumns, seedCount: seedColumns * SEED_ROWS};
  },
  build(builder, field) {
    const velocities = builder.upload('velocities', 'float32x2', field.velocities);
    const flowParameters = builder.allocate('flow-parameters', 'float32', 8);
    const ftleParameters = builder.allocate('ftle-parameters', 'float32', 4);
    const positions = builder.allocate('positions', 'float32x2', field.seedCount);
    const ftle = builder.allocate('ftle', 'float32', field.seedCount);
    builder.graph.add(
      new GPULagrangianFlowMap({
        seeds: builder.upload('seeds', 'float32x2', field.seeds).view,
        velocitiesStart: velocities.view,
        velocitiesEnd: velocities.view,
        fieldWidth: field.columns,
        fieldHeight: field.rows,
        stepCount: STEP_COUNT,
        parameters: flowParameters.view,
        output: {positions: positions.view}
      })
    );
    builder.graph.add(
      new GPUFiniteTimeLyapunovExponent({
        positions: positions.view,
        seedWidth: field.seedColumns,
        seedHeight: SEED_ROWS,
        parameters: ftleParameters.view,
        output: {ftle: ftle.view}
      })
    );
    const picture = builder.allocate('picture', 'float32', field.seedCount);
    addRowFlip(builder, 'ftle-flip', ftle, field.seedColumns, picture);
    return {
      outputs: {picture},
      update: values => {
        // Near zero the map is the instantaneous strain: keep at least MIN_TIME of integration.
        const integrationTime =
          Math.sign(values.integrationTime || 1) *
          Math.max(MIN_TIME, Math.abs(values.integrationTime));
        flowParameters.write(
          getGPULagrangianFlowMapParameterValues({
            fieldPlacement: {origin: [0, 0], cellSize: [field.cellSize, field.cellSize]},
            startTime: 0,
            timeStep: integrationTime / STEP_COUNT,
            sliceStartTime: 0,
            sliceEndTime: 1
          })
        );
        ftleParameters.write(
          getGPUFiniteTimeLyapunovExponentParameterValues({
            seedSpacingX: field.width / field.seedColumns,
            seedSpacingY: 1 / SEED_ROWS,
            integrationTime
          })
        );
      }
    };
  },
  interaction: bindAxes(parameters, {
    kind: 'time',
    hint: 'scrub integration time',
    x: 'integrationTime',
    decorate: values => ({
      readout: `integrationTime ${values.integrationTime.toFixed(2)} · ${getDirectionLabel(values.integrationTime)}`
    })
  }),
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.picture,
      columns: inputs.seedColumns,
      rows: SEED_ROWS,
      ramp: 'sequential',
      range: FTLE_RANGE,
      smoothing: true
    }
  ]
});

/** At exactly zero the map is the instantaneous strain: neither direction. */
function getDirectionLabel(integrationTime: number): string {
  if (integrationTime === 0) return 'instantaneous strain (no direction)';
  return integrationTime < 0 ? 'backward · attracting ridges' : 'forward · repelling ridges';
}
