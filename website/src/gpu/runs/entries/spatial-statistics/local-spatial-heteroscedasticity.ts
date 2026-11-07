// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalSpatialHeteroscedasticity (LOSH) over distance-band weights: how much the values around
// each point vary. Calm neighborhoods are dark; neighborhoods with mixed values are bright. The
// noise is concentrated in a band, so the band lights up as the noise rises.

import {GPULocalSpatialHeteroscedasticity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {createBandGraph, createPointField} from './shared/point-field';
import {parameters} from './local-spatial-heteroscedasticity.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local variability of the values around each point.',
  createInputs: ({random, world}) => createPointField(random, world, 700, false),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const status = builder.allocate('status', 'uint32', points.count);
    builder.graph.add(
      new GPULocalSpatialHeteroscedasticity({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        residualPower: values.residualPower === 1 ? 2 : 1,
        output: {statistic: statistic.view, status: status.view}
      })
    );
    return {
      outputs: {
        statistic,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        // Values are smooth except inside a vertical band, where noise is added.
        const mixed = Float32Array.from(points.field, (value, index) => {
          const x = points.positions[index * 2] / points.worldWidth;
          return Math.abs(x - 0.5) < 0.18
            ? value + (points.noise[index] - 0.5) * current.noise
            : value;
        });
        valueBuffer.write(mixed);
      }
    };
  },
  rebuildOn: ['residualPower', 'radius'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.statistic,
      radius: 4.5,
      range: [0, 3]
    }
  ]
});
