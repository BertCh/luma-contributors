// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULOSHResidualPermutationTest: LOSH (local spatial heteroscedasticity) plus a conditional
// permutation test of its residuals. Points whose neighborhood variability is significant are
// drawn in color. The noise sits in a vertical band, so that band is where the test fires.

import {
  getGPUPermutationParameterValues,
  GPULOSHResidualPermutationTest,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createNeighborGraph,
  createPointField,
  createProbeInputs,
  probeNeighbors
} from './shared/point-field';
import {parameters} from './losh-residual-permutation-test.parameters';

const ALTERNATIVES = ['two-sided', 'greater', 'lesser'] as const;
const MAXIMUM_PERMUTATIONS = 499;

export default defineOperationRun({
  parameters,
  caption: 'Significant local variability, from permuting the LOSH residuals.',
  createInputs: ({random, world}) => {
    const points = createPointField(random, world, 600, 0);
    return {...createProbeInputs(points), graph: createNeighborGraph(points.positions, 6)};
  },
  build(builder, points, values) {
    const {graph} = points;
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
    const parameterBuffer = builder.allocate(
      'permutation',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const status = builder.allocate('status', 'uint32', points.count);
    const exceedances = builder.allocate('exceedances', 'uint32', points.count);
    const pseudoPValues = builder.allocate('pseudo-p-values', 'float32', points.count);
    const adjustedPValues = builder.allocate('adjusted-p-values', 'float32', points.count);
    const significant = builder.allocate('significant', 'uint32', points.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULOSHResidualPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        alternative: ALTERNATIVES[values.alternative],
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {
          statistic: statistic.view,
          status: status.view,
          inference: {
            exceedances: exceedances.view,
            pseudoPValues: pseudoPValues.view,
            adjustedPValues: adjustedPValues.view,
            significant: significant.view,
            status: {overflow: overflow.view}
          }
        }
      })
    );
    return {
      outputs: {
        significant,
        statistic,
        pseudoPValues,
        bandEdges,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => {
        const mixed = Float32Array.from(points.field, (value, index) => {
          const x = points.positions[index * 2] / points.worldWidth;
          return Math.abs(x - 0.5) < 0.18
            ? value + (points.noise[index] - 0.5) * current.noise * 2
            : value;
        });
        valueBuffer.write(mixed);
        parameterBuffer.write(
          getGPUPermutationParameterValues(
            {seed: 5, permutationCount: current.permutationCount},
            {maximumPermutations: MAXIMUM_PERMUTATIONS}
          )
        );
      }
    };
  },
  rebuildOn: ['alternative'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const probe = probeNeighbors(inputs, pointer, inputs.graph.weights);
      if (probe.row < 0) return {};
      const statistic = peek(outputs.statistic)?.[probe.row];
      const pseudoP = peek(outputs.pseudoPValues)?.[probe.row];
      const isSignificant = peek(outputs.significant)?.[probe.row] === 1;
      return {
        glyphs: probe.glyphs,
        readout: `LOSH ${statistic === undefined ? '…' : statistic.toFixed(2)} · p ${
          pseudoP === undefined ? '…' : pseudoP.toFixed(3)
        }${isSignificant ? ' · significant' : ''}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    // The vertical band where noise is added to the values: the test should fire inside it.
    {layer: 'lines', context: true, segments: outputs.bandEdges, count: 2},
    {layer: 'dots', positions: outputs.positions, color: [...theme.ink, 0.4], radius: 2.5},
    // Only significant points are drawn in color: the faded mask hides the rest.
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.significant,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      radius: 5.5
    }
  ]
});
