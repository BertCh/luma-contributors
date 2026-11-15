// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GPUAdamStep,
  GPUFittingCoordinateEncoder,
  GPUFittingGradientSums,
  GPUFittingMinibatchOrder,
  getGPUAdamStepParameterValues,
  getGPUFittingMinibatchOrderParameterValues
} from '../../../src/analysis-core/index';
import {getPhilox4x32, PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {createKernelAuditContext} from '../kernel-audit/kernel-audit-context';
import {createAdamAuditProps} from '../kernel-audit/cases-fitting-kernels';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../kernel-audit/kernel-audit-wgsl';
import fixture from './fixtures/fitting-numpy-scipy.json';

it('packs per-frame Adam values and validates after float32 rounding', () => {
  expect(Array.from(getGPUAdamStepParameterValues())).toEqual(
    [0.001, 0.9, 0.999, 1e-8, 0, 0].map(Math.fround)
  );
  for (const settings of [
    {learningRate: 0},
    {learningRate: 1e100},
    {beta1: 1 - 1e-10},
    {beta2: -0.1},
    {epsilon: 1e-40},
    {tolerance: -1},
    {tolerance: Number.NaN}
  ]) {
    expect(() => getGPUAdamStepParameterValues(settings)).toThrow(/^adam-step/);
  }
  expect(getGPUAdamStepParameterValues({beta1: 0, beta2: 0, skipZeroGradient: true})[5]).toBe(1);
});

it('preserves uint32 seed words and rejects fractional or overflowing counters', () => {
  expect(
    Array.from(getGPUFittingMinibatchOrderParameterValues([0xffffffff, 0x80000000], 7))
  ).toEqual([0xffffffff, 0x80000000, 7]);
  for (const epoch of [-1, 0.5, 2 ** 32, Number.NaN]) {
    expect(() => getGPUFittingMinibatchOrderParameterValues([1, 2], epoch)).toThrow(/epoch/);
  }
});

describe('construction-time fitting limits', () => {
  it('rejects invalid shapes, model sizes, binding sizes and writable aliasing', () => {
    const context = createKernelAuditContext('fitting-rejections');
    const {view, device} = context;
    try {
      const props = createAdamAuditProps(context, 3, 8);
      expect(() => new GPUAdamStep({...props, maximumIterations: 10001})).toThrow(
        /maximumIterations.*10000/
      );
      expect(() => new GPUAdamStep({...props, gradients: view('float32', 100001)})).toThrow(
        /parameterCount.*100000/
      );
      expect(() => new GPUAdamStep({...props, loss: view('float32', 2)})).toThrow(/loss.*1/);
      expect(
        () =>
          new GPUAdamStep({
            ...props,
            output: {...props.output, secondMoment: props.output.firstMoment}
          })
      ).toThrow(/separate buffers/);
      expect(
        () =>
          new GPUFittingGradientSums({
            rowCount: -1,
            parameterCount: 1,
            gradients: view('float32', 0),
            output: {sums: view('float32', 1)}
          })
      ).toThrow(/rowCount/);
      expect(
        () =>
          new GPUFittingGradientSums({
            rowCount: 512,
            parameterCount: 100000,
            gradients: view('float32', 512 * 100000),
            output: {sums: view('float32', 100000)}
          })
      ).toThrow(/gradients byteLength.*134217728/);
    } finally {
      device.destroy();
    }
  });

  it('checks spatial conventions, feature limits and work before declaring nodes', () => {
    const context = createKernelAuditContext('encoder-rejections');
    const {view, graph, device} = context;
    try {
      const props = {
        encoding: 'spherical-harmonics' as const,
        maximumDegree: 4,
        rowCount: 1,
        coordinates: view('float32', 2),
        output: {features: view('float32', 25), validity: view('uint32', 1)}
      };
      expect(() => new GPUFittingCoordinateEncoder(props)).toThrow(/longitude-latitude|planar/);
      const spatialContext = {
        coordinateSpace: 'longitude-latitude' as const,
        metric: 'none' as const
      };
      expect(
        () => new GPUFittingCoordinateEncoder({...props, spatialContext, maximumDegree: 40})
      ).toThrow(/maximumDegree.*39/);
      expect(
        () =>
          new GPUFittingCoordinateEncoder({
            ...props,
            spatialContext,
            maximumDegree: 39,
            rowCount: 80000
          })
      ).toThrow(/encoding work.*5000000000/);
      const fourier = {
        encoding: 'fourier' as const,
        dimensions: 3,
        frequencyCount: 256,
        rowCount: 1,
        coordinates: view('float32', 3),
        frequencies: view('float32', 768),
        output: {features: view('float32', 512), validity: view('uint32', 1)}
      };
      expect(() => new GPUFittingCoordinateEncoder({...fourier, dimensions: 4})).toThrow(
        /dimensions.*3/
      );
      expect(() => new GPUFittingCoordinateEncoder({...fourier, frequencyCount: 257})).toThrow(
        /frequencyCount.*256/
      );
      expect(
        () => new GPUFittingCoordinateEncoder({...fourier, coordinates: view('float32', 2)})
      ).toThrow(/coordinates.*3/);
      const encoder = new GPUFittingCoordinateEncoder(fourier);
      const otherContext = createKernelAuditContext('different-fitting-graph');
      try {
        expect(() => encoder.getCommandNodes(otherContext.graph)).toThrow(/target graph/);
      } finally {
        otherContext.device.destroy();
      }
      expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
    } finally {
      device.destroy();
    }
  });
});

it('bounds actual reduction and sort nodes, handles empty inputs and emits fixed-order WGSL', () => {
  const context = createKernelAuditContext('fitting-source');
  const {view, graph, device} = context;
  try {
    for (const rowCount of [0, 1, 255, 256, 257, 769, 2 ** 24]) {
      const contributor = new GPUFittingGradientSums({
        id: `sum-${rowCount}`,
        rowCount,
        parameterCount: 1,
        gradients: view('float32', rowCount),
        mask: view('uint32', rowCount),
        output: {sums: view('float32', 1)}
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        expect(kernel.source).not.toMatch(/atomicAdd|var \w+: array<f32/);
      }
    }
    for (const rowCount of [0, 1, 255, 256, 257, 2 ** 24]) {
      const contributor = new GPUFittingMinibatchOrder({
        id: `order-${rowCount}`,
        rowCount,
        parameters: view('uint32', 3),
        output: {order: view('uint32', rowCount)}
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      if (rowCount === 0) expect(nodes).toHaveLength(0);
    }
    const adam = new GPUAdamStep(createAdamAuditProps(context, 3, 8));
    const sources = getGPUCommandNodeKernelSources(adam.getCommandNodes(graph));
    expect(sources).toHaveLength(adam.estimatedCommandNodeCount);
    expect(sources[0].source).toContain('history[historyOffset + completed] = currentLoss');
    expect(sources[0].source).toContain('dispatch[dispatchOffset + 0u] = 0u');
    expect(sources[1].source).toContain('getAdamSecondMomentContribution(gradient, beta2)');
    expect(sources[1].source).toContain(
      'getAdamCorrectedSecondMomentRoot(second, secondCorrection)'
    );
    expect(sources[1].source).not.toContain('sqrt(second / secondCorrection)');
    expect(sources[1].source).toContain('getAdamBiasCorrection(beta2, iteration)');
    expect(sources[1].source).toContain('bit < 14u');
    expect(sources[1].source).not.toMatch(/\bpow\s*\(/);
    for (const kernel of sources)
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('pins independent NumPy and SciPy fixtures, harmonic normalization and loss descent', () => {
  expect(fixture.quadratic.lossHistory.at(-1)).toBeLessThan(fixture.quadratic.lossHistory[0]);
  expect(fixture.logistic.lossHistory.at(-1)).toBeLessThan(fixture.logistic.lossHistory[0]);
  for (let row = 0; row < fixture.harmonics.coordinates.length; row++) {
    const values = fixture.harmonics.features.slice(row * 25, (row + 1) * 25);
    for (let degree = 0; degree <= 4; degree++) {
      const energy = values
        .slice(degree * degree, (degree + 1) ** 2)
        .reduce((sum, value) => sum + value * value, 0);
      expect(energy).toBeCloseTo((2 * degree + 1) / (4 * Math.PI), 12);
    }
  }
});

it('checks aligned binding footprints at construction and device limits before nodes', () => {
  const context = createKernelAuditContext('fitting-binding-limits');
  const {graph, view, device} = context;
  try {
    const buffer = context.buffer(128 * 1024 * 1024 + 4);
    const gradients = graph.createDataView(buffer, {
      format: 'float32',
      length: 2 ** 25,
      byteOffset: 4
    });
    expect(
      () =>
        new GPUFittingGradientSums({
          rowCount: 2 ** 24,
          parameterCount: 2,
          gradients,
          output: {sums: view('float32', 2)}
        })
    ).toThrow(/gradients byteLength.*134217728/);
    const contributor = new GPUFittingGradientSums({
      rowCount: 3,
      parameterCount: 1,
      gradients: view('float32', 3),
      output: {sums: view('float32', 1)}
    });
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 8}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(/view 0 needs 12 bytes.*limit 8/);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('matches the independent NumPy Philox fixture and pins a real key collision', () => {
  const {seed, rowCount, order, collision} = fixture.minibatch;
  const key = (row: number) =>
    getPhilox4x32([row, 0, 0, PHILOX_PURPOSES.fittingMinibatchOrder.id], [seed[0], seed[1]])[0];
  expect(
    Array.from({length: rowCount}, (_, row) => row).sort(
      (first, second) => key(first) - key(second) || first - second
    )
  ).toEqual(order);
  for (const row of collision.rows) expect(key(row)).toBe(collision.key);
});

it('pins the zero-moment first-step identity and near-one bias-correction cases', () => {
  const cases = fixture.adamBiasCorrection;
  expect(cases.some(problem => problem.iteration === 10000)).toBe(true);
  expect(cases.some(problem => problem.beta2 === Math.fround(1 - 2 ** -24))).toBe(true);
  for (const problem of cases.filter(problem => problem.iteration === 1)) {
    expect(problem.firstCorrection).toBeCloseTo(1 - problem.beta1, 15);
    expect(problem.secondCorrection).toBeCloseTo(1 - problem.beta2, 15);
    expect(problem.weight).toBeCloseTo(-Math.fround(0.05) / (1 + Math.fround(1e-8)), 14);
  }
});

it('range-controls the final Adam update before restoring its magnitude', () => {
  const context = createKernelAuditContext('adam-update-range');
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUAdamStep(createAdamAuditProps(context, 1, 1)).getCommandNodes(context.graph)
    );
    expect(sources[1].source).toContain('getAdamWeightDelta(');
    expect(sources[1].source).not.toContain(
      'parameters[parametersOffset] * (first / firstCorrection)'
    );
  } finally {
    context.device.destroy();
  }
});

it('range-controls Adam loss plateau comparison before comparing tolerance', () => {
  const context = createKernelAuditContext('adam-plateau-range');
  try {
    const sources = getGPUCommandNodeKernelSources(
      new GPUAdamStep(createAdamAuditProps(context, 1, 2)).getCommandNodes(context.graph)
    );
    expect(sources[0].source).toContain(
      'isAdamLossPlateau(currentLoss, previous, parameters[parametersOffset + 4u])'
    );
    expect(sources[0].source).not.toContain(
      'abs(currentLoss - previous) <= parameters[parametersOffset + 4u] *'
    );
  } finally {
    context.device.destroy();
  }
});
