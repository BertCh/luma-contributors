// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUQuantileLISA: tracts in the selected quantile class are the dots; the fill marks tracts where
// that class clusters significantly (conditional local join counts with permutation p-values).

import {
  getGPUPermutationParameterValues,
  GPUQuantileLISA
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {getTractFill, uploadTractMesh} from './lib/tracts';
import {findZoneAt, getTractGeometry} from './lib/zone-lookup';
import {parameters} from './quantile-lisa.parameters';

const MAXIMUM_PERMUTATIONS = 99;

export default defineOperationRun({
  parameters,
  caption:
    'Members of the chosen quantile class (tinted) and where they cluster significantly (dots).',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    const draws = Float32Array.from({length: tracts.count}, () => random() - 0.5);
    // At the default noise the tract with the highest input value sits in the top quantile.
    let restTract = 0;
    for (let row = 1; row < tracts.count; row++) {
      if (
        tracts.values[row] + draws[row] * parameters[3].default >
        tracts.values[restTract] + draws[restTract] * parameters[3].default
      ) {
        restTract = row;
      }
    }
    return {tracts, draws, restTract};
  },
  build(builder, {tracts, draws}) {
    const valueBuffer = builder.allocate('values', 'float32', tracts.count);
    const quantileParameters = builder.allocate('quantile-parameters', 'uint32', 2);
    const permutationParameters = builder.allocate('permutation-parameters', 'uint32', 4);
    const membership = builder.allocate('membership', 'uint32', tracts.count);
    const boundaries = builder.allocate('boundaries', 'float32', 2);
    const validCount = builder.allocate('valid-count', 'uint32', 1);
    const classes = builder.allocate('classes', 'uint32', tracts.count);
    const invalidQuantile = builder.allocate('invalid-quantile', 'uint32', 1);
    const exceedances = builder.allocate('exceedances', 'uint32', tracts.count);
    const pseudoPValues = builder.allocate('pseudo-p-values', 'float32', tracts.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const significant = builder.allocate('significant', 'float32x2', tracts.count);
    builder.graph.add(
      new GPUQuantileLISA({
        values: valueBuffer.view,
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        quantileParameters: quantileParameters.view,
        parameters: permutationParameters.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumNeighbors: 8,
        output: {
          membership: membership.view,
          boundaries: boundaries.view,
          validCount: validCount.view,
          classes: classes.view,
          invalidQuantile: invalidQuantile.view,
          exceedances: exceedances.view,
          pseudoPValues: pseudoPValues.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        classes,
        membership,
        pseudoPValues,
        significant,
        mesh: uploadTractMesh(builder, tracts)
      },
      update: current => {
        const quantileCount = current.numberOfQuantiles;
        quantileParameters.write(
          Uint32Array.of(quantileCount, Math.min(current.selectedQuantile, quantileCount))
        );
        permutationParameters.write(
          getGPUPermutationParameterValues({
            seed: 7,
            permutationCount: MAXIMUM_PERMUTATIONS,
            significanceLevel: current.significanceLevel
          })
        );
        valueBuffer.write(
          Float32Array.from(tracts.values, (value, index) => value + draws[index] * current.noise)
        );
      }
    };
  },
  // The pointer picks a tract; its quantile class becomes selectedQuantile, so the map shows where
  // that class clusters. Significant members are marked from the permutation p-values.
  interaction: {
    kind: 'level',
    hint: 'hover a tract',
    rest: ({toUnit, inputs}) =>
      toUnit([
        inputs.tracts.centers[inputs.restTract * 2],
        inputs.tracts.centers[inputs.restTract * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const geometry = getTractGeometry(inputs.tracts);
      const tract = findZoneAt(geometry, pointer.x, pointer.y);
      const membership = peek(outputs.membership);
      const pseudoPValues = peek(outputs.pseudoPValues);
      // Rank of the tract's input value among all tracts, as a quantile class from 1.
      const input = (row: number) => inputs.tracts.values[row] + inputs.draws[row] * values.noise;
      let below = 0;
      for (let row = 0; row < inputs.tracts.count; row++) if (input(row) < input(tract)) below++;
      const selectedQuantile = clamp(
        Math.floor((below / inputs.tracts.count) * values.numberOfQuantiles) + 1,
        1,
        values.numberOfQuantiles
      );
      let significantCount = 0;
      if (membership && pseudoPValues) {
        const marks = new Float32Array(inputs.tracts.count * 2).fill(-10);
        for (let row = 0; row < inputs.tracts.count; row++) {
          if (membership[row] === 1 && pseudoPValues[row] <= values.significanceLevel) {
            marks[significantCount * 2] = inputs.tracts.centers[row * 2];
            marks[significantCount * 2 + 1] = inputs.tracts.centers[row * 2 + 1];
            significantCount++;
          }
        }
        outputs.significant.write(marks);
      }
      return {
        values: {selectedQuantile},
        glyphs: [
          {
            shape: 'crosshair',
            at: [inputs.tracts.centers[tract * 2], inputs.tracts.centers[tract * 2 + 1]]
          }
        ],
        readout: `selectedQuantile ${selectedQuantile} of ${values.numberOfQuantiles} · ${significantCount} significant`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.membership, {
      ramp: 'sequential',
      range: [0, 1.6],
      alpha: 0.6
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    },
    {layer: 'dots', positions: outputs.significant, radius: 4, color: [1, 0.35, 0.2, 1]}
  ]
});
