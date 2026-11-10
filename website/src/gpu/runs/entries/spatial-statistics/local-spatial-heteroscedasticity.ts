// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalSpatialHeteroscedasticity (LOSH) over distance-band weights: how much the values around
// each point vary. Calm neighborhoods are dark; neighborhoods with mixed values are bright. The
// noise is concentrated in a band, so the band lights up as the noise rises.

import {GPULocalSpatialHeteroscedasticity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createBandGraph,
  createPointField,
  createProbeInputs,
  probeNeighbors
} from './shared/point-field';
import {parameters} from './local-spatial-heteroscedasticity.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local variability of the values around each point.',
  createInputs: ({random, world}) => createProbeInputs(createPointField(random, world, 700, 0)),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const bandEdges = builder.upload(
      'band-edges',
      'float32x4',
      Float32Array.of(
        points.worldWidth * 0.32,
        0,
        points.worldWidth * 0.32,
        1,
        points.worldWidth * 0.68,
        0,
        points.worldWidth * 0.68,
        1
      )
    );
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
        graph,
        bandEdges,
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
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const probe = probeNeighbors(inputs, pointer, outputs.graph.weights, values.radius);
      if (probe.row < 0) return {};
      const statistic = peek(outputs.statistic)?.[probe.row];
      return {
        glyphs: probe.glyphs,
        readout: `LOSH ${statistic === undefined ? '…' : statistic.toFixed(2)} · ${
          probe.neighborCount
        } neighbors · residualPower ${values.residualPower === 1 ? 2 : 1}`
      };
    }
  },
  draw: ({outputs}) => [
    // The vertical band where noise is added to the values.
    {layer: 'lines', context: true, segments: outputs.bandEdges, count: 2},
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.statistic,
      radius: 4.5,
      range: [0, 2]
    }
  ]
});
