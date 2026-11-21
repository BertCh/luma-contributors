// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUParticleAdvection moves particles released in a small box through the Helene wind field for
// eight frames: each particle's new position (colored by speed) and the trail back to where it was
// released. Particles are re-spawned with counter-based randomness, so the picture is the same
// every time the parameters repeat.

import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues,
  GPUParticleAdvection
} from '@luma-contributors/analysis/raster-analysis';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {parameters} from './particle-advection.parameters';
import {createWindField} from './shared/vector-field';

const PARTICLE_COUNT = 1500;
const FRAME_COUNT = 6;
/** Half the side of the release box, in world units. */
const RELEASE_RADIUS = 0.07;
const DEFAULT_RELEASE = [0.62, 0.42] as const;

export default defineOperationRun({
  parameters,
  caption: 'Particles released in the box, advected along the wind; trails show where they went.',
  createInputs: context => {
    const field = createWindField(context);
    return {
      ...field,
      release: [...DEFAULT_RELEASE] as [number, number],
      maximumSpeed: Math.max(...field.operationSpeeds)
    };
  },
  build(builder, field) {
    const worldWidth = field.columns * field.cellSize;
    const velocities = builder.upload('velocities', 'float32x2', field.velocities);
    const speedField = builder.upload('field-speeds', 'float32', field.speeds);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const wordBuffer = builder.allocate('words', 'uint32', 4);
    const positions = builder.allocate('positions', 'float32x2', PARTICLE_COUNT);
    // Where every particle was at the start of each frame: the trail is these, then `positions`.
    const frames = Array.from({length: FRAME_COUNT}, (_, frame) =>
      builder.allocate(`frame-${frame}-positions`, 'float32x2', PARTICLE_COUNT)
    );
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
          output: {
            previousPositions: frames[frame].view,
            speeds: frame === FRAME_COUNT - 1 ? speeds.view : undefined
          }
        })
      );
    }
    // One segment per particle and frame, only so the trails can be drawn. A particle that was
    // respawned jumps back to the release box: segments longer than `jumpLimit` are dropped.
    const jumpLimit = builder.allocate('jump-limit', 'float32', 1);
    const trails = frames.map((frame, index) =>
      builder.allocate(`trails-${index}`, 'float32x4', PARTICLE_COUNT)
    );
    builder.graph.add({
      getCommandNodes: graph =>
        trails.map((trail, index) =>
          createWGSLKernelNode(graph, {
            id: `particle-trails-${index}`,
            operation: 'particle-trails',
            bindings: [
              {name: 'startPoints', view: frames[index].view, type: 'f32', access: 'read'},
              {
                name: 'endPoints',
                view: (frames[index + 1] ?? positions).view,
                type: 'f32',
                access: 'read'
              },
              {name: 'limit', view: jumpLimit.view, type: 'f32', access: 'read'},
              {name: 'trail', view: trail.view, type: 'f32', access: 'read_write'}
            ],
            invocationCount: PARTICLE_COUNT,
            body: `let x0 = startPoints[startPointsOffset + index * 2u];
  let y0 = startPoints[startPointsOffset + index * 2u + 1u];
  let x1 = endPoints[endPointsOffset + index * 2u];
  let y1 = endPoints[endPointsOffset + index * 2u + 1u];
  let isJump = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) > limit[limitOffset] * limit[limitOffset];
  let missing = bitcast<f32>(0x7fc00000u | (index & 0u));
  trail[trailOffset + index * 4u] = select(x0, missing, isJump);
  trail[trailOffset + index * 4u + 1u] = select(y0, missing, isJump);
  trail[trailOffset + index * 4u + 2u] = select(x1, missing, isJump);
  trail[trailOffset + index * 4u + 3u] = select(y1, missing, isJump);`
          })
        )
    });
    return {
      outputs: {speedField, positions, speeds, trails},
      update: values => {
        const settings = {
          fieldPlacement: {origin: [0, 0], cellSize: [field.cellSize, field.cellSize]} as const,
          timeStep: values.timeStep,
          speedScale: values.speedScale,
          dropRate: values.dropRate,
          spawnBounds: [
            clamp(field.release[0] - RELEASE_RADIUS, 0, worldWidth),
            clamp(field.release[1] - RELEASE_RADIUS, 0, 1),
            clamp(field.release[0] + RELEASE_RADIUS, 0, worldWidth),
            clamp(field.release[1] + RELEASE_RADIUS, 0, 1)
          ] as const
        };
        parameterBuffer.write(
          getGPUParticleAdvectionParameterValues(settings, [field.columns, field.rows])
        );
        jumpLimit.write(
          Float32Array.of(field.maximumSpeed * values.timeStep * values.speedScale * 1.3)
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
  // The pointer is where the particles are released: they stream away from it along the wind.
  interaction: {
    kind: 'place',
    hint: 'release particles',
    rest: ({toUnit}) => toUnit(DEFAULT_RELEASE),
    resolve(pointer, {inputs, values, world}) {
      const x = clamp(pointer.x, RELEASE_RADIUS, world.width - RELEASE_RADIUS);
      const y = clamp(pointer.y, RELEASE_RADIUS, world.height - RELEASE_RADIUS);
      const moved = inputs.release[0] !== x || inputs.release[1] !== y;
      inputs.release = [x, y];
      return {
        rerun: moved,
        glyphs: [
          {
            shape: 'box',
            minimum: [x - RELEASE_RADIUS, y - RELEASE_RADIUS],
            maximum: [x + RELEASE_RADIUS, y + RELEASE_RADIUS],
            dashed: true
          },
          {shape: 'crosshair', at: [x, y]}
        ],
        readout: `release (${x.toFixed(2)}, ${y.toFixed(2)}) · timeStep ${values.timeStep}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.speedField,
      columns: inputs.columns,
      rows: inputs.rows,
      smoothing: true
    },
    ...outputs.trails.map(
      segments => ({layer: 'lines', segments, radius: 0.9, color: [1, 1, 1, 0.22]}) as const
    ),
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.speeds,
      ramp: 'sequential',
      range: [0, inputs.maximumSpeed * values.speedScale],
      radius: 2.4
    }
  ]
});
