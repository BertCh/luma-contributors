// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getGPUKernelDensityWindowOverlap} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {getGPUKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {getGPUAdaptiveBandwidthKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {
  getGPUSpaceTimeKernelDensityParameterValues,
  getGPUSpaceTimeTemporalWindowOverlap
} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUAdaptiveBandwidthKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density';
import {GPUSpaceTimeKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-space-time-kernel-density';
import {evaluateGPUSpaceTimeTemporalKernel} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {getGPUKernelDensityCanonicalBandwidthScale} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {WgslReflect} from 'wgsl_reflect';
import {getWindowMassWGSL} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-wgsl';
import {getTemporalWindowMassWGSL} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-wgsl';
import {createDensityScalarScratch} from '../../../src/gpu-spatial-analysis/point-density/density-scratch';
import {getGraphViewChunks} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

function expectRelative(actual: number, expected: number, tolerance = 2e-7) {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / expected - 1)).toBeLessThan(tolerance);
}

it('NUM-10 Gaussian overlaps retain narrow intervals and tails', () => {
  // scipy.integrate.quad(normal density, a, b); spatial expectation multiplies by erf(1/sqrt(2)).
  expectRelative(
    getGPUSpaceTimeTemporalWindowOverlap('gaussian', 1, 0, [-1e-8, 1e-8]),
    7.978845608028655e-9
  );
  expectRelative(
    getGPUKernelDensityWindowOverlap('gaussian', 1, [0, 0], [-1e-8, -1, 1e-8, 1]),
    5.447074055989346e-9
  );
  expectRelative(
    getGPUSpaceTimeTemporalWindowOverlap('gaussian', 1, 0, [9, 10]),
    1.1285122074235991e-19
  );
});

it('NUM-13 compact support caps stay positive and reflection symmetric', () => {
  // scipy.integrate.quad of (p+1)/pi * B(1/2,p+1)*(1-x*x)^(p+1/2), x=.999999..1.
  for (const [kernel, expected] of [
    ['epanechnikov', 9.60336889496883e-16],
    ['quartic', 1.646291091999591e-21]
  ] as const) {
    const positive = getGPUKernelDensityWindowOverlap(kernel, 1, [0, 0], [0.999999, -1, 1, 1]);
    expectRelative(positive, expected);
    expectRelative(
      getGPUKernelDensityWindowOverlap(kernel, 1, [0, 0], [-1, -1, -0.999999, 1]),
      expected
    );
  }
  // Exact integral: 15/16 * (4*d^3/3 - d^4 + d^5/5), d=1-.999999.
  expectRelative(
    getGPUSpaceTimeTemporalWindowOverlap('quartic', 1, 0, [0.999999, 1]),
    1.2499990626080212e-18
  );
  expectRelative(
    getGPUSpaceTimeTemporalWindowOverlap('quartic', 1, 0, [-1, -0.999999]),
    1.2499990626080212e-18
  );
});

it('FST-06 rejects unrepresentable bandwidth and grid before changing the target', () => {
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  for (const pack of [
    (target: Float32Array) =>
      getGPUKernelDensityParameterValues({bandwidth: Number.MIN_VALUE}, grid, target),
    (target: Float32Array) =>
      getGPUAdaptiveBandwidthKernelDensityParameterValues(
        {bandwidth: Number.MIN_VALUE},
        grid,
        target
      ),
    (target: Float32Array) =>
      getGPUSpaceTimeKernelDensityParameterValues(
        {bandwidth: Number.MIN_VALUE, temporalBandwidth: 1, firstSliceTime: 0, sliceInterval: 1},
        grid,
        1,
        target
      )
  ]) {
    const target = new Float32Array(16).fill(77);
    expect(() => pack(target)).toThrow(/bandwidth.*float32/);
    expect(Array.from(target)).toEqual(Array(16).fill(77));
  }
  expect(() =>
    getGPUKernelDensityParameterValues(
      {bandwidth: 1},
      {...grid, affine: [0.1, 0, 10000000, 0, -1, 1]}
    )
  ).toThrow(/x.*common.*origin/);
});

it('PCK-09 derives windows from advancing encoded slice controls', () => {
  const grid = makeExtentGrid(1, 1, [0, 0, 1, 1]);
  const settings = {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0.7, sliceInterval: 1e-9};
  expect(() => getGPUSpaceTimeKernelDensityParameterValues(settings, grid, 3)).toThrow(
    /slice.*1.*time origin/
  );
  const recentered = getGPUSpaceTimeKernelDensityParameterValues(
    {...settings, firstSliceTime: 0},
    grid,
    3
  );
  expect(recentered[13]).toBe(Math.fround(2 * Math.fround(1e-9)));
  const single = getGPUSpaceTimeKernelDensityParameterValues(settings, grid, 1);
  expect(single[12]).toBe(single[13]);
  for (const change of [
    {temporalBandwidth: Number.MIN_VALUE},
    {sliceInterval: Number.MIN_VALUE},
    {timeWindow: [0.7, 0.700000001] as const}
  ]) {
    const target = new Float32Array(16).fill(77);
    expect(() =>
      getGPUSpaceTimeKernelDensityParameterValues({...settings, ...change}, grid, 1, target)
    ).toThrow(/float32/);
    expect(Array.from(target)).toEqual(Array(16).fill(77));
  }
});

it('ENT-02 density scratch preserves structural vector chunk boundaries', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const vector = new GraphVectorView({
    id: 'foreign',
    name: 'foreign',
    format: 'float32x2',
    length: 2,
    valueLength: 4,
    stride: 2,
    byteStride: 8,
    rowByteLength: 8,
    data: [
      createTransientView(graph, 'first', 'float32x2', 1),
      createTransientView(graph, 'second', 'float32x2', 1)
    ]
  });
  // Same published shape, a different prototype, as when module formats have distinct constructors.
  const foreign = {...vector} as GraphVectorView<'float32x2'>;
  expect(foreign instanceof GraphVectorView).toBe(false);
  const scratch = createDensityScalarScratch(graph, 'scratch', foreign, 'float32');
  expect(getGraphViewChunks(scratch).map(chunk => chunk.length)).toEqual([1, 1]);
  device.destroy();
});

it('LIF-04 captures the point-density grid and nested outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'original', 'float32', 2);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    gridSize: [2, 1] as [number, number],
    bounds: [0, 0, 2, 1] as const,
    output: {values}
  };
  const density = new GPUPointDensity(props);
  props.gridSize.reverse();
  props.output.values = createTransientView(graph, 'replacement', 'float32', 2);
  expect(density.props.gridSize).toEqual([2, 1]);
  expect(density.props.output.values).toBe(values);
  expect(density.surface.values).toBe(values);
  const sources = getGPUCommandNodeKernelSources(density.getCommandNodes(graph));
  expect(sources.length).toBeGreaterThan(0);
  device.destroy();
});

it('NUM-10/NUM-13 generated overlap helpers parse together on every kernel pair', () => {
  for (const spatial of ['gaussian', 'quartic', 'epanechnikov'] as const) {
    for (const temporal of ['gaussian', 'quartic', 'epanechnikov'] as const) {
      expect(
        () =>
          new WgslReflect(
            `${getWindowMassWGSL(spatial)}\n${getTemporalWindowMassWGSL(temporal, spatial !== 'gaussian')}`
          )
      ).not.toThrow();
    }
  }
});

it('NAM-16 aliases resolve on both axes and every public math route rejects unknown names', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const common = {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 14),
    grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
    indexGridSize: [1, 1] as const,
    indexBounds: [0, 0, 1, 1] as const,
    output: {values: createTransientView(graph, 'values', 'float32', 1)}
  };
  for (const kernel of ['quartic', 'bisquare'] as const) {
    expect(new GPUKernelDensity({...common, kernel}).kernel).toBe('quartic');
    expect(new GPUAdaptiveBandwidthKernelDensity({...common, kernel}).kernel).toBe('quartic');
    const spaceTime = new GPUSpaceTimeKernelDensity({
      ...common,
      times: createTransientView(graph, `times-${kernel}`, 'float32', 3),
      sliceCount: 1,
      spatialKernel: kernel,
      temporalKernel: kernel
    });
    expect(spaceTime.spatialKernel).toBe('quartic');
    expect(spaceTime.temporalKernel).toBe('quartic');
    expectRelative(evaluateGPUSpaceTimeTemporalKernel(kernel, 1, 0.5), 0.52734375);
  }
  expect(() => evaluateGPUSpaceTimeTemporalKernel('typo' as never, 1, 0.5)).toThrow(
    /gaussian.*quartic.*epanechnikov.*bisquare/
  );
  expect(() =>
    getGPUKernelDensityWindowOverlap('typo' as never, 1, [0, 0], [-1, -1, 1, 1])
  ).toThrow(/kernel received typo/);
  // Closed form (2304/5)^(1/6), independently evaluated in the review.
  expectRelative(
    getGPUKernelDensityCanonicalBandwidthScale('quartic') /
      getGPUKernelDensityCanonicalBandwidthScale('gaussian'),
    2.77919324,
    2e-9
  );
  device.destroy();
});

it('UPT-10 declares the included population and publishes its count with eight bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const density = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    weights: createTransientView(graph, 'weights', 'float32', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
    indexGridSize: [1, 1],
    indexBounds: [-1, -1, 2, 2],
    bandwidthRule: 'scott',
    bandwidthPopulation: 'included',
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      bandwidthPopulationCount: createTransientView(graph, 'population', 'uint32', 1)
    }
  });
  const nodes = density.getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const source of sources)
    expect(() => new WgslReflect(source.source), source.id).not.toThrow();
  expect(sources.find(source => source.id.includes('bandwidth-origin-terms'))?.source).toContain(
    'all(position >= DOMAIN_MIN)'
  );
  expect(nodes.some(node => node.id.includes('bandwidth-population-count'))).toBe(true);
  expect(
    () => new GPUKernelDensity({...density.props, bandwidthPopulation: 'typo' as never})
  ).toThrow(/all-finite or included/);
  device.destroy();
});

it('CMP-05 exposes the grid-sized output band without dropping its nonzero byte offset', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const allocation = createTransientView(graph, 'allocation', 'float32', 8);
  const values = graph.createDataView(allocation.buffer, {
    format: 'float32',
    byteOffset: 8,
    length: 5
  });
  const density = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(2, 2, [0, 0, 2, 2]),
    indexGridSize: [1, 1],
    indexBounds: [0, 0, 2, 2],
    output: {values}
  });
  expect(density.outputBand.storage.values.length).toBe(4);
  expect(density.outputBand.storage.values.byteOffset).toBe(8);
  expect(density.outputBand.storage.values.buffer).toBe(values.buffer);
  device.destroy();
});

it('SCL-16 plans deterministic and atomic sums and both sides of the automatic KDE crossover', () => {
  const device = createNullWebGPUDevice();
  const limits = device.limits;
  expect(
    GPUPointDensity.getExecutionPlan({pointCount: 3, gridSize: [2, 1], weighted: true}, limits)
      .route
  ).toBe('workgroup');
  expect(
    GPUPointDensity.getExecutionPlan(
      {pointCount: 3, gridSize: [2, 1], weighted: true, sumAccumulation: 'atomic'},
      limits
    ).route
  ).toBe('atomic');
  const shape = {
    pointCount: 3,
    gridSize: [16, 16] as const,
    indexGridSize: [1, 1] as const,
    method: 'auto' as const,
    cellSize: [1, 1] as const,
    binnedPadding: 32
  };
  expect(GPUKernelDensity.getExecutionPlan({...shape, bandwidth: 6.4}, limits).route).toBe('exact');
  expect(GPUKernelDensity.getExecutionPlan({...shape, bandwidth: 6.40001}, limits).route).toBe(
    'binned'
  );
  expect(
    GPUKernelDensity.getExecutionPlan({...shape, bandwidth: 6.40001, binnedPadding: 0}, limits)
      .route
  ).toBe('exact');
  const graph = new GPUCommandGraph(device);
  const density = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(16, 16, [0, 0, 16, 16]),
    indexGridSize: [1, 1],
    indexBounds: [0, 0, 16, 16],
    method: 'auto',
    binnedPadding: 32,
    output: {values: createTransientView(graph, 'values', 'float32', 256)}
  });
  const source =
    getGPUCommandNodeKernelSources(density.getCommandNodes(graph)).find(source =>
      source.source.includes('let useBinned')
    )?.source ?? '';
  expect(source).toContain('AUTO_SUPPORT_CELLS: f32 = 32.0');
  expect(source).toContain('> AUTO_SUPPORT_CELLS');
  device.destroy();
});

it('SCL-16 numeric fixed-point cell limits match the point-density constructor on both sides', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const cells of [
    GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT,
    GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT + 1
  ]) {
    const shape = {
      pointCount: 3,
      gridSize: [cells, 1] as const,
      weighted: true,
      statistic: 'sum' as const
    };
    const plan = GPUPointDensity.getExecutionPlan(shape, device.limits);
    const construct = () =>
      new GPUPointDensity({
        positions: createTransientView(graph, `positions-${cells}`, 'float32x2', 3),
        weights: createTransientView(graph, `weights-${cells}`, 'float32', 3),
        statistic: 'sum',
        gridSize: shape.gridSize,
        bounds: [0, 0, cells, 1],
        output: {values: createTransientView(graph, `values-${cells}`, 'float32', cells)}
      });
    if (cells === GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT) {
      expect(plan.feasible).toBe(true);
      expect(construct().sumAccumulation).toBe(plan.route);
    } else {
      expect(plan.feasible).toBe(false);
      expect(construct).toThrow(/gridSize cells/);
    }
  }
  device.destroy();
});
