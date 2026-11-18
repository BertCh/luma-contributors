// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
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
  const inputValues = view('float32', 8);
  const props: GPURasterFractionalZonalDistributionProps = {
    grid: {width: 8, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    input: {id: 'input', format: 'float32', storage: {kind: 'buffer', values: inputValues}},
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
  return {graph, props, view, inputValues};
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
    for (const {id: name, source: source, constants} of sources.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source, constants), name).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
  });
}

it('RS01 rejects mismatched result shapes, aliases, and unsupported bounds', () => {
  const {props, view, inputValues} = createProps();
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
        output: {...props.output, weightSums: inputValues}
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
  for (const {id: name, source} of kernels.configurations)
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
});

it('rejects overlapping views of one imported buffer, including the per-zone status', () => {
  for (const name of ['means', 'status'] as const) {
    const {graph, props} = createProps();
    const buffer = graph.device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
    const first = importGraphBuffer(graph, 'first-output', buffer, props.output[name].format, 2);
    const second = graph.createDataView(first.buffer, {format: 'uint32', length: 2});
    expect(
      () =>
        new GPURasterFractionalZonalDistribution({
          ...props,
          output: {...props.output, [name]: first, validity: second}
        })
    ).toThrow(/(?:logical|aligned binding).*overlap/);
    buffer.destroy();
  }
});

it('rejects writable overlap with a read-only view of the same imported buffer', () => {
  const {graph, props} = createProps();
  const buffer = graph.device.createBuffer({byteLength: 256, usage: Buffer.STORAGE});
  const means = importGraphBuffer(graph, 'output-alias', buffer, 'float32', 2);
  const probabilities = graph.createDataView(means.buffer, {format: 'float32', length: 3});
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        probabilities,
        output: {...props.output, means}
      })
  ).toThrow(/(?:logical|aligned binding).*overlap/);
  buffer.destroy();
});

it('preflights aligned caller ranges before creating a resource or node', () => {
  const {graph, props} = createProps();
  const buffer = graph.createTransientBuffer({
    id: 'offset-parameters',
    byteLength: 512,
    usage: Buffer.STORAGE
  });
  const parameters = graph.createDataView(buffer, {format: 'float32', byteOffset: 252, length: 7});
  const contributor = new GPURasterFractionalZonalDistribution({...props, parameters});
  Object.defineProperty(graph.device, 'limits', {
    value: {...graph.device.limits, maxStorageBufferBindingSize: 256}
  });
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  const views = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/280 bytes/);
  expect(resources).not.toHaveBeenCalled();
  expect(views).not.toHaveBeenCalled();
});

it('preflights canonical raster scratch and maxBufferSize before declaring resources', () => {
  for (const limit of ['band', 'allocation'] as const) {
    const {graph, props, view} = createProps();
    const input =
      limit === 'band'
        ? {
            ...props.input,
            format: 'float32' as const,
            storage: {kind: 'buffer' as const, values: view('float32', 65)}
          }
        : props.input;
    const contributor = new GPURasterFractionalZonalDistribution({
      ...props,
      input,
      grid: {...props.grid, width: limit === 'band' ? 65 : 8}
    });
    Object.defineProperty(graph.device, 'limits', {
      value: {
        ...graph.device.limits,
        ...(limit === 'band' ? {maxStorageBufferBindingSize: 256} : {maxBufferSize: 128})
      }
    });
    const resources = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      limit === 'band' ? /260 bytes/ : /records.*maxBufferSize.*128/
    );
    expect(resources).not.toHaveBeenCalled();
  }
});

it('rejects mistyped scalar ports at construction', () => {
  const {props, view} = createProps();
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        probabilities: view('uint32', 3) as unknown as typeof props.probabilities
      })
  ).toThrow(/probabilities/);
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        output: {
          ...props.output,
          status: view('float32', 2) as unknown as typeof props.output.status
        }
      })
  ).toThrow(/output.status/);
});

it('validates raster calibration and nodata before canonicalization can allocate scratch', () => {
  const {props} = createProps();
  for (const value of [NaN, Infinity, 1e100]) {
    expect(
      () =>
        new GPURasterFractionalZonalDistribution({...props, input: {...props.input, offset: value}})
    ).toThrow(/^raster-fractional-zonal-distribution input.offset/);
  }
  expect(
    () =>
      new GPURasterFractionalZonalDistribution({
        ...props,
        input: {...props.input, noDataValue: Infinity}
      })
  ).toThrow(/nodata/);
});

it('rejects an uncontained empty binding before declaring resources', () => {
  const {graph, props} = createProps(1, 0, 1);
  const buffer = graph.createTransientBuffer({
    id: 'empty-at-end',
    byteLength: 256,
    usage: Buffer.STORAGE
  });
  const means = graph.createDataView(buffer, {format: 'float32', byteOffset: 256, length: 0});
  const contributor = new GPURasterFractionalZonalDistribution({
    ...props,
    output: {...props.output, means}
  });
  const resources = vi.spyOn(graph, 'createTransientBuffer');
  const views = vi.spyOn(graph, 'createDataView');
  const nodes = vi.spyOn(graph, 'addComputePass');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/binding.*(fit|buffer|range)/);
  expect(resources).not.toHaveBeenCalled();
  expect(views).not.toHaveBeenCalled();
  expect(nodes).not.toHaveBeenCalled();
});

it('accepts packed output columns sharing alignment prefixes only across separate dispatches', () => {
  const {graph, props} = createProps();
  const buffer = graph.createTransientBuffer({
    id: 'packed-distribution',
    byteLength: 2048,
    usage: Buffer.STORAGE
  });
  const offsets = [0, 256, 512, 768, 8, 264, 520, 776, 1024, 1280, 1536, 1792];
  const output = Object.fromEntries(
    Object.entries(props.output).map(([name, view], index) => [
      name,
      graph.createDataView(buffer, {
        format: view.format,
        length: view.length,
        byteOffset: offsets[index]
      })
    ])
  ) as typeof props.output;
  expect(() =>
    graph.add(new GPURasterFractionalZonalDistribution({...props, output}))
  ).not.toThrow();
});
