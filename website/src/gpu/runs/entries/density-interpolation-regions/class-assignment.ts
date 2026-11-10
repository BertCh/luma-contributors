// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUClassAssignment puts the values of every period into classes of one shared set of breaks.
// The map shows the class of each tract in the chosen period.

import {GPUClassAssignment} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './class-assignment.parameters';
import {createBreaks, createDynamics, getPeriodOwners} from './support/dynamics';
import {uploadTractMesh} from './support/geometry';

const BIN_COUNT = 32;
const STRIP = [0.06, 0.03, 0.94, 0.16] as const;
const BOTTOM_MARGIN = 0.2;

function createEffectiveBreaks(
  inputs: {values: Float32Array; override: {index: number; value: number}},
  classCount: number,
  isQuantile: boolean
): Float32Array {
  const breaks = createBreaks(inputs.values, classCount, isQuantile);
  const {index, value} = inputs.override;
  // A moved break stays between its neighbors.
  if (index >= 1 && index < classCount) {
    breaks[index] = Math.min(breaks[index + 1] - 0.005, Math.max(breaks[index - 1] + 0.005, value));
  }
  return breaks;
}

export default defineOperationRun({
  parameters,
  caption:
    'Tracts colored by class; the same breaks apply to every period. Slide a break along the histogram.',
  createInputs: ({random, world}) => {
    const dynamics = createDynamics(random, world, 12, 1, BOTTOM_MARGIN);
    const histogram = new Float32Array(BIN_COUNT);
    for (const value of dynamics.values)
      histogram[Math.min(BIN_COUNT - 1, Math.floor(value * BIN_COUNT))]++;
    return {...dynamics, histogram, override: {index: -1, value: 0}};
  },
  build(builder, inputs) {
    const {tracts, values} = inputs;
    const input = builder.upload('values', 'float32', values);
    const breaks = builder.allocate('breaks', 'float32', 10);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    const classes = builder.allocate('classes', 'uint32', values.length);
    builder.graph.add(
      new GPUClassAssignment({
        values: input.view,
        breaks: breaks.view,
        classCount: classCount.view,
        output: classes.view
      })
    );
    const mesh = uploadTractMesh(builder, tracts);
    return {
      outputs: {
        classes,
        histogram: builder.upload('histogram', 'float32', inputs.histogram),
        ...mesh
      },
      update: current => {
        classCount.write(Uint32Array.of(current.classCount));
        breaks.write(createEffectiveBreaks(inputs, current.classCount, current.breaks === 1));
        mesh.owners.write(getPeriodOwners(tracts.triangleOwners, tracts.count, current.period));
      }
    };
  },
  // The pointer's x along the histogram is a value: the nearest break of the default set moves
  // there, and the tracts are reclassified.
  interaction: {
    kind: 'level',
    hint: 'slide a break',
    rest: ({inputs, values}) => {
      const breaks = createBreaks(inputs.values, values.classCount, values.breaks === 1);
      const middle = breaks[Math.ceil(values.classCount / 2)];
      return [STRIP[0] + middle * (STRIP[2] - STRIP[0]), 0.9];
    },
    resolve(pointer, {inputs, values, world}) {
      const classCount = values.classCount;
      const isQuantile = values.breaks === 1;
      const defaults = createBreaks(inputs.values, classCount, isQuantile);
      const value = Math.min(
        1,
        Math.max(0, (pointer.x / world.width - STRIP[0]) / (STRIP[2] - STRIP[0]))
      );
      let index = 1;
      for (let candidate = 2; candidate < classCount; candidate++) {
        if (Math.abs(defaults[candidate] - value) < Math.abs(defaults[index] - value))
          index = candidate;
      }
      const isNear = Math.abs(defaults[index] - value) < 0.01;
      const previous = inputs.override;
      inputs.override = isNear ? {index: -1, value: 0} : {index, value};
      const effective = createEffectiveBreaks(inputs, classCount, isQuantile);
      const first = values.period * inputs.tracts.count;
      let memberCount = 0;
      for (let tract = 0; tract < inputs.tracts.count; tract++) {
        const tractValue = inputs.values[first + tract];
        if (tractValue > effective[index - 1] && tractValue <= effective[index]) memberCount++;
      }
      const glyphs = Array.from({length: classCount - 1}, (_, offset) => {
        const breakX = (STRIP[0] + effective[offset + 1] * (STRIP[2] - STRIP[0])) * world.width;
        return {
          shape: 'line' as const,
          from: [breakX, STRIP[1]] as const,
          to: [breakX, STRIP[3]] as const
        };
      });
      return {
        rerun: previous.index !== inputs.override.index || previous.value !== inputs.override.value,
        glyphs,
        readout: `break ${index} = ${effective[index].toFixed(2)} · class ${index}: ${memberCount} tracts`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.classes,
      triangleCount: inputs.tracts.triangleCount,
      ramp: 'sequential',
      range: [0, values.classCount - 1]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.tracts.outlineCount,
      alpha: 0.4
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.histogram,
      range: 'auto',
      rectangle: STRIP,
      color: [0.62, 0.72, 0.92, 0.9]
    }
  ]
});
