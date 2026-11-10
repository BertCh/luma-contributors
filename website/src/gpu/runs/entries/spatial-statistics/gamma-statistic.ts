// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGammaStatistic: the sum over neighbor pairs of a pair function (cross-product,
// squared-difference or absolute-difference) of their values. The map shows the values; hovering a
// point lights its neighbor pairs and the part of the sum they contribute. The bar is the
// statistic the GPU wrote.

import {GPUGammaStatistic} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createBandGraph,
  createNeighborGlyphs,
  createPointField,
  createProbeInputs,
  getMixedValue,
  getNeighborRows,
  mixNoise,
  pickProbeRow
} from './shared/point-field';
import {parameters} from './gamma-statistic.parameters';

const BAND_RADIUS = 0.08;
const OPERATIONS = ['cross-product', 'squared-difference', 'absolute-difference'] as const;
const OBSERVED_BAR = [0.25, 0.04, 0.48, 0.22] as const;
const EXPECTED_BAR = [0.52, 0.04, 0.75, 0.22] as const;

function getPairValue(operation: number, first: number, second: number): number {
  if (operation === 0) return first * second;
  return operation === 1 ? (first - second) ** 2 : Math.abs(first - second);
}

/**
 * Sum of the pair function over the neighbor slots as observed, and as expected when the values
 * are shuffled over the points (every ordered pair equally likely). Their ratio to the GPU's
 * statistic is the operation's normalization.
 */
function getPairSums(
  values: Float32Array,
  weights: {neighbors: Uint32Array; offsets: Uint32Array},
  operation: number
) {
  let observedSum = 0;
  for (let row = 0; row < values.length; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      observedSum += getPairValue(operation, values[row], values[weights.neighbors[slot]]);
    }
  }
  let total = 0;
  for (let first = 0; first < values.length; first++) {
    for (let second = 0; second < values.length; second++) {
      if (first !== second) total += getPairValue(operation, values[first], values[second]);
    }
  }
  const expectedSum = (total / (values.length * (values.length - 1))) * weights.neighbors.length;
  return {observedSum, expectedSum};
}

export default defineOperationRun({
  parameters,
  caption: 'Gamma statistic: a pair function summed over every neighbor pair.',
  createInputs: ({random, world}) => {
    const points = createPointField(random, world, 700);
    return {
      ...createProbeInputs(points),
      graph: createBandGraph(points.positions, BAND_RADIUS),
      sums: {key: '', observedSum: 0, expectedSum: 0}
    };
  },
  build(builder, points, values) {
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const statistic = builder.allocate('statistic', 'float32', 1);
    const observedBar = builder.allocate('observed-bar', 'float32', 1);
    const expectedBar = builder.allocate('expected-bar', 'float32', 1);
    builder.graph.add(
      new GPUGammaStatistic({
        weights: uploadSpatialWeights(builder, points.graph.weights),
        values: valueBuffer.view,
        operation: OPERATIONS[values.operation],
        output: {statistic: statistic.view}
      })
    );
    return {
      outputs: {
        valueBuffer,
        statistic,
        observedBar,
        expectedBar,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => valueBuffer.write(mixNoise(points.field, points.noise, current.noise))
    };
  },
  rebuildOn: ['operation'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const floor = OBSERVED_BAR[1] * world.height;
      const baseline = {
        shape: 'line',
        from: [OBSERVED_BAR[0] * world.width, floor],
        to: [EXPECTED_BAR[2] * world.width, floor]
      } as const;
      const statistic = peek(outputs.statistic)?.[0];
      const key = `${values.operation}:${values.noise}`;
      if (inputs.sums.key !== key) {
        const mixed = mixNoise(inputs.field, inputs.noise, values.noise);
        Object.assign(inputs.sums, getPairSums(mixed, inputs.graph.weights, values.operation), {
          key
        });
      }
      let expected = NaN;
      if (statistic !== undefined && inputs.sums.observedSum > 0) {
        expected = (inputs.sums.expectedSum * statistic) / inputs.sums.observedSum;
        // Both bars share one scale, set by the larger, so only their ratio is drawn.
        const scale = 1 / (1.3 * Math.max(statistic, expected));
        outputs.observedBar.write(Float32Array.of(statistic * scale));
        outputs.expectedBar.write(Float32Array.of(expected * scale));
      }
      const summary = Number.isFinite(expected)
        ? `Γ ${statistic!.toFixed(3)} vs ${expected.toFixed(3)} expected`
        : 'Γ …';
      const row = pickProbeRow(inputs, pointer);
      if (row < 0) return {glyphs: [baseline], readout: summary};
      const own = getMixedValue(inputs, row, values.noise);
      let pairSum = 0;
      const neighbors = getNeighborRows(inputs.graph.weights, row);
      for (const neighbor of neighbors) {
        pairSum += getPairValue(
          values.operation,
          own,
          getMixedValue(inputs, neighbor, values.noise)
        );
      }
      const share =
        statistic === undefined
          ? ''
          : ` (${((100 * pairSum) / inputs.sums.observedSum).toFixed(2)}% of Γ)`;
      return {
        glyphs: [
          baseline,
          ...createNeighborGlyphs(inputs.positions, inputs.graph.weights, row),
          {
            shape: 'ring',
            at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
            radius: BAND_RADIUS,
            dashed: true
          }
        ],
        readout: `${neighbors.length} pairs${share} · ${summary}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    // An ink ring under every dot keeps the low end of the ramp readable on both backgrounds.
    {
      layer: 'dots',
      positions: outputs.positions,
      radius: 7.5,
      color: [...theme.ink, 0.9]
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.valueBuffer,
      radius: 5.5,
      range: [-0.35, 0.9]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.observedBar,
      range: [0, 1],
      rectangle: OBSERVED_BAR,
      color: [...theme.categories[1], 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.expectedBar,
      range: [0, 1],
      rectangle: EXPECTED_BAR,
      color: [...theme.ink, 0.45]
    }
  ]
});
