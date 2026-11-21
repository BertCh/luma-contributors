// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSegregation,
  getGPUSegregationLayout
} from '../../../src/gpu-spatial-analysis/segregation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeSpatialAuditKernel} from '../spatial-validation/spatial-design-audit.test-utils';
import {computeSegregationOracle} from './segregation-oracle';

it.each([
  [[8, NaN, 2, 8], 1, 0],
  [[8, 0, 2, 8], 0, 0.8],
  [[8, 2, 2, 8], 0, 0.6]
] as const)('UST-10 excludes incomplete demographic units: %j', (counts, invalid, expected) => {
  const reference = computeSegregationOracle({
    unitCount: 2,
    groupCount: 2,
    counts,
    weights: null,
    missingCountPolicy: 'exclude-unit'
  });
  // D=1/2 sum |x/X-y/Y|. A sole complete unit has D=0;
  // two units [8,0],[2,8] give .8 and [8,2],[2,8] give .6.
  const dissimilarity = reference.indices[getGPUSegregationLayout(2).dissimilarity];
  expect(Number.isFinite(dissimilarity)).toBe(true);
  expect(dissimilarity).toBeCloseTo(expected, 12);
  expect(reference.invalidUnitCount).toBe(invalid);
});

it('UST-10 sanitation reports each invalid unit once and preserves observed zeros', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `missing-${serial++}`, format, length);
  try {
    const props = {
      groupCount: 2,
      groupCounts: view('float32', 4),
      missingCountPolicy: 'exclude-unit' as const,
      output: {
        indices: view('float32', getGPUSegregationLayout(2).stride),
        invalidUnitCount: view('uint32', 1)
      }
    };
    const nodes = new GPUSegregation(props).getCommandNodes(graph);
    const [kernel] = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id === 'segregation-sanitize-counts')
    );
    expect(kernel).toBeDefined();
    const invalidUnitCount = new Uint32Array(1),
      groupCounts = new Float32Array(4);
    executeSpatialAuditKernel(kernel, {
      rawCounts: Float32Array.of(8, NaN, 2, 8),
      groupCounts,
      invalidUnitCount
    });
    expect(Array.from(groupCounts)).toEqual([-1, -1, 2, 8]);
    expect(Array.from(invalidUnitCount)).toEqual([1]);
    expect(
      () => new GPUSegregation({...props, missingCountPolicy: 'bad' as 'exclude-unit'})
    ).toThrow(/zero.*exclude-unit/);
  } finally {
    device.destroy();
  }
});

it('S4 omits excluded weighted targets from smoothed populations and local results', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `weighted-missing-${serial++}`, format, length);
  try {
    const contributor = new GPUSegregation({
      groupCount: 2,
      groupCounts: view('float32', 4),
      missingCountPolicy: 'exclude-unit',
      spatialForm: 'smoothed-population',
      scales: [
        {offsets: view('uint32', 3), neighbors: view('uint32', 2), weights: view('float32', 2)}
      ],
      output: {
        indices: view('float32', 13),
        localEnvironment: view('float32', 4),
        localEntropy: view('float32', 2)
      }
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const environment = kernels.find(kernel => kernel.id === 'segregation-scale-0-environment')!;
    const local = kernels.find(kernel => kernel.id === 'segregation-scale-0-local')!;
    const values = {
      parameters: Float32Array.of(1, 0.5),
      groupCounts: Float32Array.of(-1, -1, 2, 8),
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 0),
      weights: Float32Array.of(1, 1),
      environment: new Float32Array(7),
      scaleCursor: Uint32Array.of(0xffffffff),
      totals: Float32Array.of(2, 8, 10),
      localEnvironment: new Float32Array(4),
      localEntropy: new Float32Array(2)
    };
    executeSpatialAuditKernel(environment, values);
    // The missing target gets no neighbor population; the sole retained population is [2,8].
    expect(Array.from(values.environment)).toEqual([0, 0, 2, 8, 0, 10, 0]);
    executeSpatialAuditKernel(local, {...values, groupCounts: values.environment});
    expect(Array.from(values.localEnvironment.slice(0, 2))).toEqual([NaN, NaN]);
    expect(values.localEntropy[0]).toBeNaN();
    expect(Number.isFinite(values.localEntropy[1])).toBe(true);
    expect(Math.abs(values.localEntropy[1] - 0.5004024235381879)).toBeLessThan(1e-6);
    // Mutation check: dropping the target-exclusion branch must invalidate the environment fixture.
    const mutated = {
      ...environment,
      source: environment.source.replace(
        /if \(groupCounts\[groupCountsOffset \+ index \* K\] < 0\.0\) \{[\s\S]*?return;\n  \}/,
        ''
      )
    };
    expect(mutated.source).not.toBe(environment.source);
    values.environment.fill(0);
    values.scaleCursor[0] = 0xffffffff;
    executeSpatialAuditKernel(mutated, values);
    expect(Array.from(values.environment)).not.toEqual([0, 0, 2, 8, 0, 10, 0]);
  } finally {
    device.destroy();
  }
});
