// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUInequality} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {getGPUInequalityParameterValues} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {computeInequalityOnCPU} from '../../../src/dataframe-analysis/composite-indicators/cpu-inequality-operation';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {INEQUALITY_INPUT, INEQUALITY_EXPECTED} from './inequality-output-oracles-fixture';

it('CPU inequality outputs match independent NumPy indices and decomposition', () => {
  const result = computeInequalityOnCPU(INEQUALITY_INPUT);
  for (const name of Object.keys(INEQUALITY_EXPECTED) as (keyof typeof INEQUALITY_EXPECTED)[])
    expectLiteralRows(result[name === 'gini' ? 'weightedGini' : name], INEQUALITY_EXPECTED[name]);
  // Mutation guard: publishing the Gini port as Hoover must fail on zone [1,2,6].
  expect(() => expectLiteralRows(result.weightedGini, INEQUALITY_EXPECTED.hoover)).toThrow();
});

it('CPU Atkinson retains precision for large weights one float32 step above epsilon one', () => {
  const result = computeInequalityOnCPU({
    values: Float32Array.of(1, 3),
    weights: Float32Array.of(2 ** 126, 2 ** 126),
    zoneIds: Uint32Array.of(0, 0),
    zoneCount: 1,
    parameters: getGPUInequalityParameterValues({epsilon: 1 + 2 ** -23})
  });
  // Equal relative weights give the same Decimal result as [1, 1, 3, 3] with unit weights.
  expect(result.atkinson).toEqual(Float32Array.of(0.133974611790977));
});

const ATKINSON_FIXTURES: {
  name: string;
  values: Float32Array;
  weights?: Float32Array;
  epsilon: number;
  expected: number;
}[] = [
  {
    name: 'underflowed weighted income with positive mean',
    values: Float32Array.of(0, 2 ** -80),
    weights: Float32Array.of(2 ** -80, 2 ** -80),
    epsilon: 3,
    expected: 1
  },
  {
    name: 'all-zero income with tiny weights',
    values: Float32Array.of(0, 0),
    weights: Float32Array.of(2 ** -80, 2 ** -80),
    epsilon: 3,
    expected: NaN
  },
  {
    name: 'adjacent epsilon',
    values: new Float32Array([1, 1, 3, 3]),
    epsilon: 1 + 2 ** -23,
    expected: 0.133974611790977
  },
  {
    name: 'zero income below one',
    values: new Float32Array([0, ...new Array(512).fill(1), ...new Array(512).fill(3)]),
    epsilon: 0.999,
    expected: 0.673332558
  },
  {
    name: 'large weights below one (R1)',
    values: Float32Array.of(0, 1 / 256, 1),
    weights: Float32Array.of(2 ** 110, 2 ** 126, 2 ** 126),
    epsilon: 0.999,
    expected: 0.8759559391035893
  },
  {
    name: 'zero income at one',
    values: new Float32Array([0, 1, 1, 3, 3]),
    epsilon: 1,
    expected: 1
  },
  {
    name: 'large weights at epsilon one',
    values: Float32Array.of(1 / 1024, 1),
    weights: Float32Array.of(2 ** 126, 2 ** 126),
    epsilon: 1,
    expected: 0.9375609756097561
  },
  {
    name: 'zero income above one',
    values: new Float32Array([0, 1, 1, 3, 3]),
    epsilon: 1 + 2 ** -23,
    expected: 1
  },
  {
    name: 'extreme positive ratio',
    values: new Float32Array([1e-30, 1e30]),
    epsilon: 1 - 2 ** -8,
    expected: 1
  }
];

it.each(ATKINSON_FIXTURES)('Atkinson reference shader arithmetic: $name', fixture => {
  const mode = 'reference';
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = createTransientView(graph, 'values', 'float32', fixture.values.length);
    const parameters = createTransientView(graph, 'parameters', 'float32', 4);
    const weights = fixture.weights
      ? createTransientView(graph, 'weights', 'float32', fixture.weights.length)
      : undefined;
    const contributor = new GPUInequality({
      id: 'near-one',
      mode,
      values,
      weights,
      parameters,
      zoneIds: createTransientView(graph, 'zones', 'uint32', fixture.values.length),
      zoneCount: 1,
      output: {atkinson: createTransientView(graph, 'atkinson', 'float32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    const views = nodes.flatMap(
      node =>
        node.resources?.flatMap(resource =>
          'buffer' in resource && resource.buffer instanceof GraphDataView ? [resource.buffer] : []
        ) ?? []
    );
    const findView = (suffix: string) =>
      views.find(view => view.buffer.id === `near-one-${suffix}`)!;
    const memory = createCPUGraphMemory();
    memory.write(values, fixture.values);
    if (weights && fixture.weights) memory.write(weights, fixture.weights);
    memory.write(parameters, getGPUInequalityParameterValues({epsilon: fixture.epsilon}));
    memory.write(findView('sorted-zone-keys'), new Uint32Array(fixture.values.length));
    memory.write(
      findView('zone-sorted-rows'),
      Uint32Array.from({length: fixture.values.length}, (_, row) => row)
    );
    const stats = findView('stats');
    executeCommandNodesOnCPU(
      [
        nodes.find(node => node.id === 'near-one-zone-stats')!,
        nodes.find(node => node.id === 'near-one-atkinson-zero-boundary')!
      ],
      memory
    );
    // Decimal oracle in inequality-output-oracles.oracle.py; float32 kernel tolerance 5e-4.
    expectLiteralRows([memory.readFloat32(stats)[4]], [fixture.expected], 5e-4);
    expectLiteralRows(
      computeInequalityOnCPU({
        values: fixture.values,
        weights: fixture.weights,
        zoneIds: new Uint32Array(fixture.values.length),
        zoneCount: 1,
        parameters: getGPUInequalityParameterValues({epsilon: fixture.epsilon})
      }).atkinson,
      [fixture.expected],
      5e-4
    );
  } finally {
    device.destroy();
  }
});
