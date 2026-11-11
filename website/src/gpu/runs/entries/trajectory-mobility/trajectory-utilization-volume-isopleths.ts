// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUUtilizationVolumeIsopleths on the Brownian bridge density of one San Francisco Bay vessel
// track. Each cell shows how much probability mass lies in cells at least as dense; the outline
// is the home range at `level`: the smallest set of densest cells holding that share.

import {
  getGPUUtilizationVolumeIsoplethsParameterValues,
  GPUUtilizationVolumeIsopleths
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {addBrownianBridge, createHomeRangeInputs, type HomeRangeInputs} from './_lib/home-range';
import {readFloat32, readUint32} from './_lib/movement';
import {parameters} from './trajectory-utilization-volume-isopleths.parameters';

type Inputs = HomeRangeInputs;
type Outputs = {
  segments: RunBuffer;
  density: RunBuffer;
  volume: RunBuffer;
  inside: RunBuffer;
};

export default defineOperationRun({
  parameters,
  caption:
    'Bright cells are the core of the density (volume near 0), dark the fringe. The outline holds `level` of the probability mass.',
  createInputs: ({world}) => createHomeRangeInputs(world),
  build(builder, inputs) {
    const bridge = addBrownianBridge(builder, inputs);
    const parameterBuffer = builder.allocate('isopleth-parameters', 'float32', 4);
    const cellCount = inputs.columns * inputs.rows;
    const volume = builder.allocate('volume', 'float32', cellCount);
    const inside = builder.allocate('inside', 'uint32', cellCount);
    builder.graph.add(
      new GPUUtilizationVolumeIsopleths({
        density: bridge.density.view,
        parameters: parameterBuffer.view,
        output: {volume: volume.view, inside: inside.view}
      })
    );
    const outputs: Outputs = {
      segments: bridge.segments,
      density: bridge.density,
      volume,
      inside
    };
    return {
      outputs,
      update: current => {
        bridge.write(current.motionVariance, 0.005);
        parameterBuffer.write(
          getGPUUtilizationVolumeIsoplethsParameterValues({level: current.level})
        );
      }
    };
  },
  // The core brightness (one minus the volume) and the outline of the chosen isopleth.
  async afterRun({inputs, outputs, upload}) {
    const [volume, inside] = await Promise.all([
      readFloat32(outputs.volume),
      readUint32(outputs.inside)
    ]);
    const {columns, rows, world} = inputs;
    const core = Float32Array.from(volume, value => (value > 0 ? 1 - value : 0));
    const outline: number[] = [];
    const isInside = (column: number, row: number) =>
      column >= 0 &&
      column < columns &&
      row >= 0 &&
      row < rows &&
      inside[row * columns + column] === 1;
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        if (!isInside(column, row)) continue;
        const left = (column / columns) * world.width;
        const right = ((column + 1) / columns) * world.width;
        const top = 1 - row / rows;
        const bottom = 1 - (row + 1) / rows;
        if (!isInside(column - 1, row)) outline.push(left, bottom, left, top);
        if (!isInside(column + 1, row)) outline.push(right, bottom, right, top);
        if (!isInside(column, row - 1)) outline.push(left, top, right, top);
        if (!isInside(column, row + 1)) outline.push(left, bottom, right, bottom);
      }
    }
    return {
      core: upload('core', 'float32', core),
      outlineCount: outline.length / 4,
      outline: upload(
        'outline',
        'float32x4',
        Float32Array.from(outline.length ? outline : [0, 0, 0, 0])
      )
    };
  },
  // Level: the volume of the cell under the pointer is the level at which that cell joins the
  // home range.
  // Level: the pointer's x across the canvas is `level`, from the 5% core to the 99% range; the
  // outline and the cells inside follow it.
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'level',
    hint: 'level across',
    x: 'level',
    decorate: (values, _pointer, {inputs, outputs, peek}) => {
      const inside = peek(outputs.inside);
      let insideCount = 0;
      for (let index = 0; inside && index < inside.length; index++) {
        insideCount += inside[index] === 1 ? 1 : 0;
      }
      const area =
        (insideCount * inputs.world.width * inputs.metersPerUnit ** 2) /
        (inputs.columns * inputs.rows) /
        1e6;
      return {
        readout: `level ${values.level.toFixed(2)} (${values.level < 0.6 ? 'core' : 'range'}) · ${area.toFixed(0)} km² inside`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    ...(extra
      ? [
          {
            layer: 'raster' as const,
            values: extra.core,
            columns: inputs.columns,
            rows: inputs.rows,
            ramp: 'sequential' as const,
            range: [0, 1] as const,
            bands: 10,
            alpha: 0.9
          },
          {
            layer: 'lines' as const,
            segments: extra.outline,
            count: extra.outlineCount,
            radius: 2,
            color: [1, 1, 1, 1] as const
          }
        ]
      : [])
  ]
});
