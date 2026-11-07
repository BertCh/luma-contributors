// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUParticleAdvection moves particles through a seeded swirl field for one frame: each particle's
// new position (bright, colored by speed) and where it came from (muted). Particles are re-spawned
// with counter-based randomness, so the picture is the same every time the parameters repeat.

import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues,
  GPUParticleAdvection
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './particle-advection.parameters';
import {createSwirlField} from './shared/vector-field';

const PARTICLE_COUNT = 6000;
const FRAME_COUNT = 10;

export default defineOperationRun({
  parameters,
  caption: 'Particles one step downstream (bright) from where they were (muted).',
  createInputs: context => createSwirlField(context),
  build(builder, field) {
    const worldWidth = field.columns * field.cellSize;
    const velocities = builder.upload('velocities', 'float32x2', field.velocities);
    const speedField = builder.upload('field-speeds', 'float32', field.speeds);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const wordBuffer = builder.allocate('words', 'uint32', 4);
    const positions = builder.allocate('positions', 'float32x2', PARTICLE_COUNT);
    const previousPositions = builder.allocate('previous-positions', 'float32x2', PARTICLE_COUNT);
    const speeds = builder.allocate('speeds', 'float32', PARTICLE_COUNT);
    const state = {
      positions: positions.view,
      ages: builder.allocate('ages', 'uint32', PARTICLE_COUNT).view,
      generations: builder.allocate('generations', 'uint32', PARTICLE_COUNT).view
    };
    const resetWords = builder.allocate('reset-words', 'uint32', 4);
    // One encoding is one frame, so the operation is added once per frame; only the first resets.
    for (let frame = 0; frame < FRAME_COUNT; frame++) {
      builder.graph.add(
        new GPUParticleAdvection({
          id: `particle-advection-${frame}`,
          velocities: velocities.view,
          fieldWidth: field.columns,
          fieldHeight: field.rows,
          parameters: parameterBuffer.view,
          uint32Parameters: frame === 0 ? resetWords.view : wordBuffer.view,
          state,
          output:
            frame === 0
              ? {previousPositions: previousPositions.view}
              : {speeds: frame === FRAME_COUNT - 1 ? speeds.view : undefined}
        })
      );
    }
    return {
      outputs: {speedField, positions, previousPositions, speeds},
      update: values => {
        const settings = {
          fieldExtent: [0, 0, field.cellSize, field.cellSize] as const,
          timeStep: values.timeStep,
          speedScale: values.speedScale,
          dropRate: values.dropRate,
          spawnBounds: [0, 0, worldWidth, 1] as const
        };
        parameterBuffer.write(
          getGPUParticleAdvectionParameterValues(settings, [field.columns, field.rows])
        );
        // Every run starts from a fresh, deterministic particle set.
        resetWords.write(
          getGPUParticleAdvectionUint32ParameterValues({
            seed: 7,
            frame: 0,
            maximumAge: 0,
            reset: true
          })
        );
        wordBuffer.write(
          getGPUParticleAdvectionUint32ParameterValues({seed: 7, frame: 1, maximumAge: 0})
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.speedField,
      columns: inputs.columns,
      rows: inputs.rows,
      smoothing: true
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.previousPositions,
      radius: 1.6,
      color: [1, 1, 1, 0.35]
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.speeds,
      ramp: 'sequential',
      radius: 2
    }
  ]
});
