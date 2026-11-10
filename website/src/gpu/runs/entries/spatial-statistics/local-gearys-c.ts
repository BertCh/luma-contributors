// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalGeary over distance-band weights. Small values mark points whose neighbors have
// similar values (clusters); large values mark points that differ from their neighbors.

import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPULocalGeary
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createBandGraph,
  createPointField,
  createProbeInputs,
  mixNoise,
  probeNeighbors
} from './shared/point-field';
import {parameters} from './local-gearys-c.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local Geary per point: low where neighbors agree, high where they differ.',
  createInputs: ({random, world}) => createProbeInputs(createPointField(random, world, 700, 0)),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const statistic = builder.allocate('statistic', 'float32', points.count);
    builder.graph.add(
      new GPULocalGeary({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        parameters: parameterBuffer.view,
        islands: values.islands === 1 ? 'keep' : 'exclude',
        output: {statistic: statistic.view}
      })
    );
    return {
      outputs: {
        statistic,
        graph,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        valueBuffer.write(mixNoise(points.field, points.noise, current.noise));
        parameterBuffer.write(getGPUSpatialAutocorrelationParameterValues());
      }
    };
  },
  rebuildOn: ['islands', 'radius'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const probe = probeNeighbors(inputs, pointer, outputs.graph.weights, values.radius);
      if (probe.row < 0) return {};
      const statistic = peek(outputs.statistic)?.[probe.row];
      if (probe.neighborCount === 0) {
        return {glyphs: probe.glyphs, readout: `no neighbors within radius ${values.radius}`};
      }
      const meaning = statistic === undefined ? '' : statistic < 1 ? 'agree' : 'differ';
      return {
        glyphs: probe.glyphs,
        readout: `c ${statistic === undefined ? '…' : statistic.toFixed(2)} · ${probe.neighborCount} neighbors ${meaning}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.statistic,
      radius: 4.5,
      range: [0, 4]
    }
  ]
});
