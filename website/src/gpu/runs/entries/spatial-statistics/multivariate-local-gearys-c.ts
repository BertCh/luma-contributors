// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultivariateLocalGeary: one local Geary statistic over several variables at once, averaged
// across the variables. Variables share one spatial pattern; the noise slider breaks it
// independently in each variable, so more variables give a more reliable picture of the cluster.

import {GPUMultivariateLocalGeary} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createBandGraph,
  createInterleavedValues,
  createPointField,
  createProbeInputs,
  getNeighborRows,
  getVariableValue,
  probeNeighbors
} from './shared/point-field';
import {parameters} from './multivariate-local-gearys-c.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Local Geary averaged over several variables.',
  createInputs: ({random, world}) => createProbeInputs(createPointField(random, world, 700, 0)),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const variableCount = values.variableCount;
    const valueBuffer = builder.allocate('values', 'float32', points.count * variableCount);
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const status = builder.allocate('status', 'uint32', points.count);
    const contributions = builder.allocate('contributions', 'float32', variableCount);
    builder.graph.add(
      new GPUMultivariateLocalGeary({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        variableCount,
        output: {statistic: statistic.view, status: status.view}
      })
    );
    return {
      outputs: {
        statistic,
        graph,
        contributions,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current =>
        valueBuffer.write(createInterleavedValues(points.field, variableCount, current.noise))
    };
  },
  rebuildOn: ['variableCount', 'radius'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const probe = probeNeighbors(inputs, pointer, outputs.graph.weights, values.radius);
      const shares = new Float32Array(outputs.contributions.length);
      if (probe.row >= 0) {
        // Each variable's squared differences to the neighbors, as a share of the total.
        let total = 0;
        for (let variable = 0; variable < shares.length; variable++) {
          for (const neighbor of getNeighborRows(outputs.graph.weights, probe.row)) {
            shares[variable] +=
              (getVariableValue(inputs.field, probe.row, variable, values.noise) -
                getVariableValue(inputs.field, neighbor, variable, values.noise)) **
              2;
          }
          total += shares[variable];
        }
        if (total > 0)
          for (let variable = 0; variable < shares.length; variable++) shares[variable] /= total;
      }
      outputs.contributions.write(shares);
      if (probe.row < 0) return {};
      const statistic = peek(outputs.statistic)?.[probe.row];
      return {
        glyphs: probe.glyphs,
        readout: `c ${statistic === undefined ? '…' : statistic.toFixed(2)} · ${
          shares.length
        } variables · ${probe.neighborCount} neighbors`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.statistic,
      radius: 4.5,
      range: [0, 1.4]
    },
    // Share of the hovered point's total squared difference that each variable contributes.
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.contributions,
      range: [0, 1],
      rectangle: [0.03, 0.04, 0.25, 0.2],
      color: [...theme.categories[1], 0.95]
    }
  ]
});
