// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURasterFractionalZonalDistribution,
  type GPURasterFractionalZonalDistributionProps
} from '../../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';
import fixture from '../../oracles/fixtures/fractional-zonal-distribution.json';

function createProps(capacity = 8, zoneCount = 2, probabilityCount = 3) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `distribution-${identifier++}`, format, length);
  const props: GPURasterFractionalZonalDistributionProps = {
    grid: {width: 8, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    input: {id: 'input', format: 'float32', storage: {kind: 'buffer', values: view('float32', 8)}},
    zoneCount,
    coverage: {
      zoneIds: view('uint32', capacity),
      cellIds: view('uint32', capacity),
      fractions: view('float32', capacity),
      zoneValidity: view('uint32', zoneCount),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    },
    probabilities: view('float32', probabilityCount),
    output: {
      means: view('float32', zoneCount),
      populationVariances: view('float32', zoneCount),
      sampleVariances: view('float32', zoneCount),
      effectiveSampleSizes: view('float32', zoneCount),
      weightSums: view('float32', zoneCount),
      squaredWeightSums: view('float32', zoneCount),
      coveredArea: view('float32', zoneCount),
      validArea: view('float32', zoneCount),
      quantiles: view('float32', zoneCount * probabilityCount),
      quantileValidity: view('uint32', zoneCount * probabilityCount),
      validity: view('uint32', zoneCount),
      status: view('uint32', zoneCount)
    }
  };
  return {graph, props, view};
}

for (const [capacity, zoneCount, probabilityCount] of [
  [8, 2, 3],
  [65536, 65536, 32],
  [1, 0, 1]
]) {
  it(`RS01 parses bounded distribution kernels: ${capacity} tuples/${zoneCount} zones/${probabilityCount} quantiles`, () => {
    const {graph, props} = createProps(capacity, zoneCount, probabilityCount);
    const sources = getParsedKernelSources(
      new GPURasterFractionalZonalDistribution(props).getCommandNodes(graph)
    );
    expect(sources.size).toBeGreaterThan(5);
    for (const [name, source] of sources) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), name).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
  });
}

it('RS01 rejects mismatched result shapes, aliases, and unsupported bounds', () => {
  const {props, view} = createProps();
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        output: {...props.output, quantiles: view('float32', 2)}
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        output: {...props.output, means: props.output.populationVariances}
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        output: {...props.output, weightSums: props.input.storage.values}
      })
  ).toThrow();
  expect(
    () => new GPURasterFractionalZonalDistribution({...props, probabilities: view('float32', 33)})
  ).toThrow();
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        coverage: {...props.coverage, zoneIds: view('uint32', 65537)}
      })
  ).toThrow();
  expect(() => new GPURasterFractionalZonalDistribution({...props, zoneCount: -1})).toThrow();
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        grid: {...props.grid, affine: [1, 1, 0, 1, 1, 0]}
      })
  ).toThrow();
});

it('RS01 fixture provenance distinguishes direct library calls from composed support', () => {
  expect(fixture.versions.exactextract).toBe('0.3.0');
  expect(fixture.reference.direct).toContain('exactextract.exact_extract');
  expect(fixture.reference.sampleVariance).toContain('numpy.cov');
  const equalValues = fixture.cases.find(
    entry => entry.name === 'equal-value-distinct-cell-support'
  )!;
  expect(equalValues.expected.quantiles).toEqual([3, 3, 3, 3, 3, 3, 3]);
  expect(equalValues.expected.effectiveSampleSizes).toBeGreaterThan(2);
  const missing = fixture.cases.find(entry => entry.name === 'nodata-retains-covered-area')!;
  expect(missing.expected.coveredArea).toBeGreaterThan(missing.expected.validArea);
});

it('RS01 independent weighting kernels stay within storage and memory budgets', () => {
  const {graph, props, view} = createProps();
  const contributor = new GPURasterFractionalZonalDistribution({
    ...props,
    observationWeights: view('float32', 8),
    cellAreas: view('float32', 8),
    quantileMethod: 'inverse-cdf'
  });
  const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
  for (const [name, source] of kernels)
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
});
