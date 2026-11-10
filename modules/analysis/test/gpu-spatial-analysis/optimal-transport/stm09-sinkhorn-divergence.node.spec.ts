// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSinkhornDivergenceParameterValues,
  GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH,
  GPUSinkhornDivergence,
  type GPUSinkhornDivergenceProps
} from '../../../src/gpu-spatial-analysis/optimal-transport/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/stm09-sinkhorn-pot.json';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeSinkhornDivergenceReference} from './stm09-sinkhorn-reference';

let viewCount = 0;

function createContributor(
  graph: GPUCommandGraph,
  width: number,
  height: number,
  overrides: Partial<GPUSinkhornDivergenceProps> = {},
  outputs: 'all' | 'minimal' = 'all'
) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `sinkhorn-node-view-${viewCount++}`, format, length);
  const cellCount = width * height;
  return new GPUSinkhornDivergence({
    id: overrides.id ?? `sinkhorn-${viewCount++}`,
    grid: makeGPURasterMetadata({width, height}),
    source: view('float32', cellCount),
    target: view('float32', cellCount),
    parameters: view('float32', GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH),
    output:
      outputs === 'minimal'
        ? {divergence: view('float32', 1)}
        : {
            divergence: view('float32', 1),
            components: view('float32', 3),
            marginalError: view('float32', 1),
            status: {
              converged: view('uint32', 1),
              iterationLimitReached: view('uint32', 1),
              iterationCount: view('uint32', 1),
              invalidCount: view('uint32', 1),
              approximation: view('uint32', 1)
            }
          },
    ...overrides
  });
}

it('GPUSinkhornDivergence estimates its command node count as an upper bound, exactly per iteration', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'sinkhorn-nodes'});
  for (const [width, height, maximumIterations, outputs] of [
    [8, 8, 1, 'minimal'],
    [16, 12, 10, 'all'],
    [64, 64, 100, 'all'],
    [5, 5, 250, 'minimal']
  ] as const) {
    const contributor = createContributor(graph, width, height, {maximumIterations}, outputs);
    const nodeCount = contributor.getCommandNodes(graph).length;
    expect(nodeCount).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const single = createContributor(graph, width, height, {maximumIterations: 1}, outputs);
    expect(nodeCount - single.getCommandNodes(graph).length).toBe(7 * (maximumIterations - 1));
  }
});

it('GPUSinkhornDivergence rejects out-of-range capacities and shapes with named limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'sinkhorn-limits'});
  const make = (overrides: Partial<GPUSinkhornDivergenceProps>, width = 8, height = 8) =>
    createContributor(graph, width, height, {id: 'sinkhorn', ...overrides});
  expect(() => make({maximumIterations: 0})).toThrow(
    'sinkhorn maximumIterations must be an integer in [1, 1000]'
  );
  expect(() => make({maximumIterations: 1001})).toThrow(
    'sinkhorn maximumIterations must be an integer in [1, 1000]'
  );
  expect(() => make({blurRadius: 0})).toThrow(
    'sinkhorn blurRadius must be an integer in [1, 2048]'
  );
  expect(() => make({}, 2049, 2)).toThrow(
    'sinkhorn grid.width and grid.height must be at most 2048'
  );
  expect(() =>
    make({source: createTransientView(graph, 'sinkhorn-short-source', 'float32', 63)})
  ).toThrow('sinkhorn source and target must hold grid.width * grid.height rows');
  expect(() =>
    make({parameters: createTransientView(graph, 'sinkhorn-short-parameters', 'float32', 5)})
  ).toThrow('sinkhorn parameters must hold 6 float32 values');
});

it('getGPUSinkhornDivergenceParameterValues packs the layout and rejects bad values', () => {
  const values = getGPUSinkhornDivergenceParameterValues({
    cellWidth: 2,
    cellHeight: 3,
    epsilon: 0.5,
    epsilonStart: 4,
    scalingFactor: 0.5
  });
  expect(Array.from(values)).toEqual([2, 3, 0.5, 1e-4, 4, 0.5].map(Math.fround));
  const settings = {cellWidth: 2, cellHeight: 3, epsilon: 0.5};
  for (const name of Object.keys(settings) as (keyof typeof settings)[]) {
    expect(() => getGPUSinkhornDivergenceParameterValues({...settings, [name]: 0})).toThrow(
      `getGPUSinkhornDivergenceParameterValues ${name} must be positive and finite`
    );
  }
  expect(() => getGPUSinkhornDivergenceParameterValues({...settings, epsilonStart: 0.1})).toThrow(
    'epsilonStart must be at least epsilon'
  );
  expect(() => getGPUSinkhornDivergenceParameterValues({...settings, scalingFactor: 1.5})).toThrow(
    'scalingFactor must be in (0, 1]'
  );
  expect(() => getGPUSinkhornDivergenceParameterValues(settings, new Float32Array(2))).toThrow(
    'target must hold 6 elements'
  );
});

it('the binary64 reference gives the closed forms and is consistent with the POT fixture', () => {
  // Two point masses at squared distance 13: S = 13 whatever epsilon is (no oracle needed).
  const source = new Array(25).fill(0);
  const target = new Array(25).fill(0);
  source[0] = 2;
  target[2 * 5 + 3] = 1;
  const deltas = computeSinkhornDivergenceReference({
    width: 5,
    height: 5,
    cellWidth: 1,
    cellHeight: 1,
    source,
    target,
    epsilon: 3
  });
  expect(deltas.divergence).toBeCloseTo(13, 12);
  const fixture = decodeOracleFixture<{
    cases: {
      name: string;
      width: number;
      height: number;
      cellWidth: number;
      cellHeight: number;
      epsilon: number;
      source: number[];
      target: number[];
      divergence: number;
    }[];
  }>(fixtureJson);
  // The two binary64 implementations (this dense log-domain code and POT's coupling) agree.
  for (const entry of fixture.cases.filter(item => item.width * item.height <= 100)) {
    const reference = computeSinkhornDivergenceReference(entry, 1e-11);
    expect(Math.abs(reference.divergence - entry.divergence), entry.name).toBeLessThan(1e-7);
  }
  expect(fixture.cases.map(item => item.name)).toContain('identical-8x8-epsilon-2');
});
