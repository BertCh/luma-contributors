// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeRestrictedRegimeRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {getRegimeMomentStride} from '../../../src/gpu-spatial-analysis/spatial-regression/regime-moments';
import {ROBUST_RESTRICTION_FIXTURE as fixture} from './robust-restriction-fixture';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {
  GPURestrictedRegimeRegressionOperation,
  type RestrictedRegimeRegressionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation';
import {CPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {constructRegressionOperationContributor} from './regression-operation-test-utils';

function createInput(shift = 0) {
  return {
    predictors: Float32Array.from(
      fixture.predictors,
      (value, slot) => value + (slot % 2 === 0 ? shift : 0)
    ),
    response: Float32Array.from(fixture.response),
    regimes: Uint32Array.from(fixture.regimes),
    mask: Uint32Array.from(fixture.mask)
  };
}

function createKernel() {
  return {
    offsets: Uint32Array.from(fixture.offsets),
    neighbors: Uint32Array.from(fixture.neighbors),
    weights: Float32Array.from(fixture.weights)
  };
}

function expectRows(
  actual: Float32Array | Uint32Array,
  expected: readonly (number | null)[],
  tolerance = 2e-6
) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, slot) => {
    if (value === null) expect(actual[slot], `slot ${slot}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[slot]), `slot ${slot}`).toBe(true);
      expect(
        Math.abs(actual[slot] - value),
        `slot ${slot}: ${actual[slot]} vs ${value}`
      ).toBeLessThanOrEqual(tolerance * Math.max(1, Math.abs(value)));
    }
  });
}

const options = {predictorCount: 2, regimeCount: 3, common: [false, true, false]};

for (const method of ['white', 'hc1', 'hac'] as const) {
  it(`matches dense NumPy ${method} covariance and SciPy Wald tails on every output`, () => {
    const result = computeRestrictedRegimeRegressionOnCPU(createInput(), {
      ...options,
      robust:
        method === 'hac'
          ? {type: 'hac', kernelWeights: createKernel()}
          : {type: 'white', smallSample: method === 'hc1'}
    });
    for (const port of [
      'coefficients',
      'covariance',
      'fitted',
      'chow',
      'stability',
      'status'
    ] as const)
      expectRows(result[port], fixture.expected[port]);
    expect(result.robust).toBeDefined();
    for (const port of ['covariance', 'chow', 'stability', 'status'] as const)
      expectRows(result.robust![port], fixture.expected[method][port]);
    // The heteroskedastic fixture discriminates HC0, HC1, HAC and pooled inference.
    expect(Math.abs(result.robust!.chow[0] - result.stability[9])).toBeGreaterThan(1);
    const shifted = computeRestrictedRegimeRegressionOnCPU(createInput(1e6), {
      ...options,
      robust:
        method === 'hac'
          ? {type: 'hac', kernelWeights: createKernel()}
          : {type: 'white', smallSample: method === 'hc1'}
    });
    expectRows(shifted.robust!.status, [0]);
    expectRows(shifted.robust!.chow, fixture.expected[method].chow);
    expectRows(shifted.robust!.stability, fixture.expected[method].stability);
    expectRows(shifted.fitted, fixture.expected.fitted);
    // Raw covariance must transform with intercept <- intercept - shift * commonSlope.
    const transform = Array.from({length: 7}, (_, row) =>
      Array.from(
        {length: 7},
        (_, column) => Number(row === column) - (row < 3 && column === 3 ? 1e6 : 0)
      )
    );
    const covariance = fixture.expected[method].covariance;
    const transformed = transform.flatMap(row =>
      transform.map(column =>
        row.reduce(
          (sum, left, first) =>
            sum +
            column.reduce(
              (innerSum, right, second) => innerSum + left * right * covariance[first * 7 + second],
              0
            ),
          0
        )
      )
    );
    expectRows(shifted.robust!.covariance, transformed, 3e-6);
  });
}

it('rejects incomplete, missing-diagonal and asymmetric kernels and suppresses every dependent output', () => {
  for (const defect of [
    'overflow',
    'candidateOverflow',
    'converged',
    'diagonal',
    'asymmetric'
  ] as const) {
    const kernel = createKernel();
    if (defect === 'diagonal') kernel.weights[0] = 0.5;
    if (defect === 'asymmetric') kernel.weights[1] = 0.125;
    const result = computeRestrictedRegimeRegressionOnCPU(createInput(), {
      ...options,
      robust: {
        type: 'hac',
        kernelWeights: {
          ...kernel,
          status:
            defect.endsWith('Overflow') || defect === 'overflow' || defect === 'converged'
              ? {[defect]: Uint32Array.of(defect === 'converged' ? 0 : 1)}
              : undefined
        }
      }
    });
    expectRows(result.robust!.status, [2]);
    expect(result.robust!.covariance.every(Number.isNaN)).toBe(true);
    expectRows(result.robust!.chow, [null, 0, null]);
    expectRows(result.robust!.stability, [
      null,
      0,
      null,
      null,
      0,
      null,
      null,
      0,
      null,
      null,
      0,
      null
    ]);
    expectRows(result.status, [0]);
    expectRows(result.coefficients, fixture.expected.coefficients);
  }
});

it('reports non-positive contrast covariance, unavailable fits, and no varying coefficients explicitly', () => {
  // Symmetric, nonnegative, unit-diagonal K can still be indefinite. K=I+100 adjacency
  // gives a negative tested contrast variance on this fixed fixture.
  const kernel = createKernel();
  kernel.weights = Float32Array.from(kernel.weights, value => (value === 1 ? 1 : 100 * value));
  const indefinite = computeRestrictedRegimeRegressionOnCPU(createInput(), {
    ...options,
    robust: {type: 'hac', kernelWeights: kernel}
  });
  expectRows(indefinite.robust!.status, [3]);
  expect(indefinite.robust!.covariance.every(Number.isNaN)).toBe(true);
  expectRows(indefinite.robust!.chow, [null, 0, null]);
  const unavailable = computeRestrictedRegimeRegressionOnCPU(
    {...createInput(), mask: new Uint32Array(33)},
    {...options, robust: {type: 'white'}}
  );
  expectRows(unavailable.status, [1]);
  expectRows(unavailable.robust!.status, [1]);
  expect(unavailable.robust!.covariance.every(Number.isNaN)).toBe(true);
  const pooled = computeRestrictedRegimeRegressionOnCPU(createInput(), {
    ...options,
    common: [true, true, true],
    robust: {type: 'white'}
  });
  expectRows(pooled.robust!.status, [0]);
  expectRows(pooled.robust!.chow, [null, 0, null]);
  expectRows(pooled.robust!.stability, [
    null,
    0,
    null,
    null,
    0,
    null,
    null,
    0,
    null,
    null,
    0,
    null
  ]);
});

it('accepts retained-edge counts and a supplied converged flag of one', () => {
  const result = computeRestrictedRegimeRegressionOnCPU(createInput(), {
    ...options,
    robust: {
      type: 'hac',
      kernelWeights: {
        ...createKernel(),
        status: {
          count: Uint32Array.of(fixture.neighbors.length),
          requiredCount: Uint32Array.of(fixture.neighbors.length),
          iterationCount: Uint32Array.of(2),
          converged: Uint32Array.of(1)
        }
      }
    }
  });
  for (const port of ['covariance', 'chow', 'stability', 'status'] as const)
    expectRows(result.robust![port], fixture.expected.hac[port]);
});

it('keeps legacy outputs bitwise unchanged when robust inference is requested', () => {
  const classical = computeRestrictedRegimeRegressionOnCPU(createInput(), options);
  const robust = computeRestrictedRegimeRegressionOnCPU(createInput(), {
    ...options,
    robust: {type: 'white'}
  });
  expect(classical.robust).toBeUndefined();
  for (const port of [
    'coefficients',
    'covariance',
    'fitted',
    'chow',
    'stability',
    'status'
  ] as const)
    expect(Array.from(robust[port])).toEqual(Array.from(classical[port]));
  expect(getRegimeMomentStride(2)).toBe(12); // 3 means + 6 triangular products + 3 deviation sums.
});

for (const model of ['commonIntercept', 'pivot'] as const)
  for (const method of ['white', 'hc1', 'hac'] as const) {
    it(`matches independent ${model} ${method} inference`, () => {
      const input = createInput();
      if (model === 'pivot')
        input.predictors = Float32Array.from(
          input.predictors,
          (value, slot) => value + (slot % 2 === 1 ? 1e6 : 0)
        );
      const result = computeRestrictedRegimeRegressionOnCPU(input, {
        ...options,
        common: [true, true, false],
        robust:
          method === 'hac'
            ? {type: 'hac', kernelWeights: createKernel()}
            : {type: 'white', smallSample: method === 'hc1'}
      });
      for (const port of [
        'coefficients',
        'covariance',
        'fitted',
        'chow',
        'stability',
        'status'
      ] as const)
        expectRows(result[port], fixture[model][port]);
      for (const port of ['covariance', 'chow', 'stability', 'status'] as const)
        expectRows(result.robust![port], fixture[model][method][port]);
    });
  }

it('constructs both operation forms with matching robust ports and snapshots nested options', () => {
  const names = {
    coefficients: 'coefficients',
    covariance: 'covariance',
    fitted: 'fitted',
    chow: 'chow',
    stability: 'stability',
    status: 'status',
    robust: {
      covariance: 'robustCovariance',
      chow: 'robustChow',
      stability: 'robustStability',
      status: 'robustStatus'
    }
  };
  const props: RestrictedRegimeRegressionOperationProps = {
    ...options,
    rowCount: 33,
    predictors: 'predictors',
    response: 'response',
    regimes: 'regimes',
    mask: 'mask',
    robust: {
      type: 'hac',
      kernelWeights: {
        offsets: 'kernelOffsets',
        neighbors: 'kernelNeighbors',
        weights: 'kernelWeights',
        status: {overflow: 'kernelOverflow'}
      }
    },
    output: names
  };
  const operation = new GPURestrictedRegimeRegressionOperation(props);
  const cpuOperation = new CPURestrictedRegimeRegressionOperation(props);
  expect(cpuOperation.contract.outputs).toEqual(operation.contract.outputs);
  const shapes = {
    predictors: {format: 'float32' as const, length: 66},
    response: {format: 'float32' as const, length: 33},
    regimes: {format: 'uint32' as const, length: 33},
    mask: {format: 'uint32' as const, length: 33},
    kernelOffsets: {format: 'uint32' as const, length: 34},
    kernelNeighbors: {format: 'uint32' as const, length: fixture.neighbors.length},
    kernelWeights: {format: 'float32' as const, length: fixture.weights.length},
    kernel_overflow: {format: 'uint32' as const, length: 1}
  };
  props.common = [true];
  props.output.robust!.covariance = 'wrong';
  if (props.robust?.type === 'hac') props.robust.kernelWeights.offsets = 'wrong';
  const contributor = constructRegressionOperationContributor(
    operation,
    shapes
  ) as GPURestrictedRegimeRegression;
  expect(contributor.props.common).toEqual([false, true, false]);
  expect(contributor.props.output.robust!.covariance.length).toBe(49);
  expect(contributor.props.robust?.type).toBe('hac');
});

it('checks robust output pairing, kernel shape/metadata, aliases and graph construction without executing GPU code', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floatView = (length: number) =>
      createTransientView(graph, `float-${serial++}`, 'float32', length);
    const uintView = (length: number) =>
      createTransientView(graph, `uint-${serial++}`, 'uint32', length);
    const robustOutput = {
      covariance: floatView(49),
      chow: floatView(3),
      stability: floatView(12),
      status: uintView(1)
    };
    const props = {
      ...options,
      common: [...options.common],
      predictors: floatView(66),
      response: floatView(33),
      regimes: uintView(33),
      mask: uintView(33),
      robust: {type: 'white' as const},
      output: {
        coefficients: floatView(7),
        covariance: floatView(49),
        fitted: floatView(33),
        chow: floatView(3),
        stability: floatView(12),
        status: uintView(1),
        robust: robustOutput
      }
    };
    const contributor = new GPURestrictedRegimeRegression(props);
    props.common[0] = true;
    props.robust.type = 'white';
    props.output.robust.chow = floatView(1);
    expect(contributor.props.common).toEqual([false, true, false]);
    expect(contributor.props.output.robust!.chow.length).toBe(3);
    expect(contributor.getCommandNodes(graph).some(node => node.id.endsWith('robust-tests'))).toBe(
      true
    );
    const valid = {
      ...props,
      common: [false, true, false],
      output: {...props.output, robust: {...robustOutput, chow: floatView(3)}}
    };
    expect(() => new GPURestrictedRegimeRegression({...valid, robust: undefined})).toThrow(
      /robust/
    );
    expect(
      () =>
        new GPURestrictedRegimeRegression({...valid, output: {...valid.output, robust: undefined}})
    ).toThrow(/robust/);
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...valid,
          output: {
            ...valid.output,
            robust: {...valid.output.robust, covariance: valid.output.covariance}
          }
        })
    ).toThrow(/overlap|disjoint|share/i);
    const kernel = {
      offsets: uintView(34),
      neighbors: uintView(fixture.neighbors.length),
      weights: floatView(fixture.weights.length)
    };
    const hac = new GPURestrictedRegimeRegression({
      ...valid,
      id: 'hac-test',
      robust: {
        type: 'hac',
        kernelWeights: {...kernel, status: {overflow: uintView(1), candidateOverflow: uintView(1)}}
      }
    });
    const nodes = hac.getCommandNodes(graph);
    const lossNodes = nodes.filter(node => node.id.startsWith('hac-test-robust-kernel-status-'));
    const statusViews =
      hac.props.robust?.type === 'hac' ? hac.props.robust.kernelWeights.status : undefined;
    for (const view of [statusViews?.overflow, statusViews?.candidateOverflow]) {
      expect(view).toBeDefined();
      expect(
        lossNodes.some(node =>
          node.resources?.some(
            resource =>
              'buffer' in resource && resource.buffer === view && resource.usage === 'storage-read'
          )
        )
      ).toBe(true);
    }
    const lossOutput = lossNodes
      .at(-1)
      ?.resources?.find(
        resource => 'buffer' in resource && resource.usage === 'storage-read-write'
      );
    expect(lossOutput && 'buffer' in lossOutput).toBe(true);
    const covarianceNode = nodes.find(node => node.id === 'hac-test-robust-covariance');
    expect(
      covarianceNode?.resources?.some(
        resource =>
          'buffer' in resource &&
          lossOutput &&
          'buffer' in lossOutput &&
          resource.buffer === lossOutput.buffer &&
          resource.usage === 'storage-read'
      )
    ).toBe(true);
    expect(
      covarianceNode?.resources?.some(
        resource =>
          'buffer' in resource &&
          resource.buffer === hac.props.output.robust?.status &&
          resource.usage === 'storage-read-write'
      )
    ).toBe(true);
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...valid,
          robust: {type: 'hac', kernelWeights: {...kernel, offsets: uintView(33)}}
        })
    ).toThrow(/robust.kernelWeights/);
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...valid,
          robust: {
            type: 'hac',
            kernelWeights: {
              ...kernel,
              information: {rowCount: 33, transform: 'row', symmetric: true}
            }
          }
        })
    ).toThrow(/unstandardized/);
  } finally {
    device.destroy();
  }
});
