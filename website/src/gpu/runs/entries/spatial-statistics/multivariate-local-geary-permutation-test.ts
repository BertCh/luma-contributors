// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultivariateLocalGearyPermutationTest: classifies each point as a cluster, a negative
// association or not significant, from the multivariate local Geary statistic and shuffles of
// the neighbors. The exact strategy handles up to 256 rows, so this run uses 240 points.

import {GPUMultivariateLocalGearyPermutationTest} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createInterleavedValues,
  createNeighborGraph,
  createPointField,
  createProbeInputs,
  probeNeighbors
} from './shared/point-field';
import {parameters} from './multivariate-local-geary-permutation-test.parameters';

const CLASS_NAMES = ['not significant', 'cluster', 'negative association'];
const TAILS = ['equal-tail', 'esda-min-tail'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Multivariate local Geary classes: cluster, negative association, not significant.',
  createInputs: ({random, world}) => {
    const points = createPointField(random, world, 240, 0);
    return {...createProbeInputs(points), graph: createNeighborGraph(points.positions, 6)};
  },
  build(builder, points, values) {
    const {graph} = points;
    const variableCount = values.variableCount;
    const valueBuffer = builder.allocate('values', 'float32', points.count * variableCount);
    const statistic = builder.allocate('statistic', 'float32', points.count);
    const pValues = builder.allocate('p-values', 'float32', points.count);
    const classes = builder.allocate('classes', 'uint32', points.count);
    builder.graph.add(
      new GPUMultivariateLocalGearyPermutationTest({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        variableCount,
        maximumPermutations: values.maximumPermutations,
        seed: [1, 2],
        tail: TAILS[values.tail],
        output: {statistic: statistic.view, pValues: pValues.view, classes: classes.view}
      })
    );
    return {
      outputs: {
        classes,
        statistic,
        pValues,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current =>
        valueBuffer.write(createInterleavedValues(points.field, variableCount, current.noise))
    };
  },
  rebuildOn: ['variableCount', 'tail', 'maximumPermutations'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const probe = probeNeighbors(inputs, pointer, inputs.graph.weights);
      if (probe.row < 0) return {};
      const pValue = peek(outputs.pValues)?.[probe.row];
      const classCode = peek(outputs.classes)?.[probe.row];
      return {
        glyphs: probe.glyphs,
        readout: `${classCode === undefined ? '…' : (CLASS_NAMES[classCode] ?? `class ${classCode}`)} · p ${
          pValue === undefined ? '…' : pValue.toFixed(3)
        } · ${values.variableCount} variables`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.classes,
      ramp: 'categorical',
      // Class 0 (not significant) is muted so only the significant classes stand out.
      categories: [theme.muted, theme.categories[1], theme.categories[0]],
      radius: 6
    }
  ]
});
