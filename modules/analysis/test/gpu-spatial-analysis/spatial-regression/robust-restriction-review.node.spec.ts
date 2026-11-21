// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {computeRestrictedRegimeRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {ROBUST_RESTRICTION_FIXTURE as fixture} from './robust-restriction-fixture';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPURestrictedRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression';
import {GPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation';
import {CPURestrictedRegimeRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation';
import {constructRegressionOperationContributor} from './regression-operation-test-utils';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {getRegressionTileLayout} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';

function createHost() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const floatView = (length: number) =>
    createTransientView(graph, `review-float-${serial++}`, 'float32', length);
  const uintView = (length: number) =>
    createTransientView(graph, `review-uint-${serial++}`, 'uint32', length);
  const props = {
    predictors: floatView(8),
    response: floatView(8),
    regimes: uintView(8),
    predictorCount: 1,
    regimeCount: 2,
    common: [false, true],
    robust: {type: 'white' as const},
    output: {
      coefficients: floatView(3),
      covariance: floatView(9),
      fitted: floatView(8),
      chow: floatView(3),
      stability: floatView(9),
      status: uintView(1),
      robust: {
        covariance: floatView(9),
        chow: floatView(3),
        stability: floatView(9),
        status: uintView(1)
      }
    }
  };
  return {device, graph, floatView, uintView, props};
}

const regimes = Uint32Array.of(0, 1, 0, 1, 0, 1, 0, 1);
const response = Float32Array.of(0, 1, 4, 9, 3, 12, 10, 10);

it('R3 uses restricted residual degrees when no unrestricted residual degrees remain', () => {
  const result = computeRestrictedRegimeRegressionOnCPU(
    {
      predictors: Float32Array.of(0, 0, 1, 1),
      regimes: Uint32Array.of(0, 1, 0, 1),
      response: Float32Array.of(0, 1, 4, 9)
    },
    {predictorCount: 1, regimeCount: 2, common: [false, true], robust: {type: 'white'}}
  );
  // Restricted beta=[-1,2,6], e=[1,-1,-1,1], residual df=1; unrestricted residual df=0.
  expect(result.status[0]).toBe(1);
  expect(result.robust!.status[0]).toBe(0);
  expect(Array.from(result.robust!.covariance)).toEqual([
    0.75, 0.25, -0.5, 0.25, 0.75, -0.5, -0.5, -0.5, 1
  ]);
  expect(result.robust!.chow[0]).toBeCloseTo(9, 5);
  expect(result.robust!.chow[2]).toBeCloseTo(0.0026997960632601883, 8);
  expect(result.coefficients.every(Number.isNaN)).toBe(true);
});

it('R2 rejects the numerically zero all-ones HAC sandwich before equilibration', () => {
  const result = computeRestrictedRegimeRegressionOnCPU(
    {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
    {
      predictorCount: 1,
      regimeCount: 2,
      common: [false, true],
      robust: {
        type: 'hac',
        kernelWeights: {
          offsets: Uint32Array.from({length: 9}, (_, row) => row * 8),
          neighbors: Uint32Array.from({length: 64}, (_, slot) => slot % 8),
          weights: new Float32Array(64).fill(1)
        }
      }
    }
  );
  // K=11': X'diag(e)Kdiag(e)X = (X'e)(X'e)' = 0 by the OLS normal equations.
  expect(result.robust!.status[0]).toBe(3);
  expectNaNCovariance(result.robust!.covariance, 3);
  expect(Array.from(result.robust!.chow)).toEqual([NaN, 0, NaN]);
});

it('R9 crosses the row-tile boundary and obeys independent duplicate-sample HC0 scaling', () => {
  expect(getRegressionTileLayout(33).tileCount).toBe(1);
  expect(getRegressionTileLayout(66)).toEqual({tileRowCount: 64, tileCount: 2});
  const result = computeRestrictedRegimeRegressionOnCPU(
    {
      predictors: Float32Array.from([...fixture.predictors, ...fixture.predictors]),
      response: Float32Array.from([...fixture.response, ...fixture.response]),
      regimes: Uint32Array.from([...fixture.regimes, ...fixture.regimes]),
      mask: Uint32Array.from([...fixture.mask, ...fixture.mask])
    },
    {predictorCount: 2, regimeCount: 3, common: [false, true, false], robust: {type: 'white'}}
  );
  expect(result.robust!.status[0]).toBe(0);
  fixture.expected.white.covariance.forEach((value, slot) =>
    expect(result.robust!.covariance[slot]).toBeCloseTo(value / 2, 6)
  );
  expect(result.robust!.covariance[0]).toBeCloseTo(0.00296013184, 8);
  expect(result.robust!.chow[0]).toBeCloseTo(28.17795671174808, 5);
  expect(result.robust!.chow[1]).toBe(4);
  expect(result.robust!.chow[2]).toBeCloseTo(0.000011478741611841643, 9);
});

it('R8 snapshots spatial-regime props and output bindings before caller mutation', () => {
  const host = createHost();
  try {
    const counts = host.uintView(2);
    const props = {
      id: 'review-spatial',
      predictorCount: 1,
      regimeCount: 2,
      predictors: host.floatView(8),
      response: host.floatView(8),
      regimes: host.uintView(8),
      output: {
        coefficients: host.floatView(4),
        fitted: host.floatView(8),
        status: host.uintView(2),
        counts
      }
    };
    const contributor = new GPUSpatialRegimeRegression(props);
    props.regimeCount = 3;
    props.output.counts = host.uintView(3);
    expect(contributor.props.regimeCount).toBe(2);
    expect(contributor.props.output.counts).toBe(counts);
    const fit = contributor
      .getCommandNodes(host.graph)
      .find(node => node.id === 'review-spatial-fit');
    expect(fit).toBeDefined();
    expect(fit!.workload?.maximumInvocationCount).toBe(2);
  } finally {
    host.device.destroy();
  }
});

it('R7 omits explicit undefined optional status ports in both operation constructors', () => {
  const props = {
    rowCount: 8,
    predictorCount: 1,
    regimeCount: 2,
    common: [false, true],
    predictors: 'predictors',
    regimes: 'regimes',
    response: 'response',
    robust: {
      type: 'hac' as const,
      kernelWeights: {
        offsets: 'kernelOffsets',
        neighbors: 'kernelNeighbors',
        weights: 'kernelWeights',
        status: {overflow: undefined}
      }
    },
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      fitted: 'fitted',
      chow: 'chow',
      status: 'status',
      robust: {
        covariance: 'robustCovariance',
        chow: 'robustChow',
        stability: 'robustStability',
        status: 'robustStatus'
      }
    }
  };
  for (const Operation of [
    GPURestrictedRegimeRegressionOperation,
    CPURestrictedRegimeRegressionOperation
  ]) {
    const operation = new Operation(props);
    expect(operation.inputs).not.toHaveProperty('kernel_overflow');
  }
  const host = createHost();
  try {
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...host.props,
          robust: {
            type: 'hac',
            kernelWeights: {
              offsets: host.uintView(9),
              neighbors: host.uintView(8),
              weights: host.floatView(8),
              status: {overflow: undefined}
            }
          }
        })
    ).not.toThrow();
    const result = computeRestrictedRegimeRegressionOnCPU(
      {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
      {
        predictorCount: 1,
        regimeCount: 2,
        common: [false, true],
        robust: {
          type: 'hac',
          kernelWeights: {
            offsets: Uint32Array.from({length: 9}, (_, row) => row),
            neighbors: Uint32Array.from({length: 8}, (_, row) => row),
            weights: new Float32Array(8).fill(1),
            status: {overflow: undefined}
          }
        }
      }
    );
    expect(result.robust!.status[0]).toBe(0);
  } finally {
    host.device.destroy();
  }
});

it('R7 requires exactly one word for each supplied kernel status through direct and operation forms', async () => {
  const host = createHost();
  try {
    const kernel = {
      offsets: host.uintView(9),
      neighbors: host.uintView(8),
      weights: host.floatView(8),
      status: {overflow: host.uintView(2)}
    };
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...host.props,
          robust: {type: 'hac', kernelWeights: kernel}
        })
    ).toThrow(/status.*length|status.*one/);
    expect(() =>
      computeRestrictedRegimeRegressionOnCPU(
        {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
        {
          predictorCount: 1,
          regimeCount: 2,
          common: [false, true],
          robust: {
            type: 'hac',
            kernelWeights: {
              offsets: Uint32Array.from({length: 9}, (_, row) => row),
              neighbors: Uint32Array.from({length: 8}, (_, row) => row),
              weights: new Float32Array(8).fill(1),
              status: {overflow: new Uint32Array(2)}
            }
          }
        }
      )
    ).toThrow(/status.*length/);
    const operationProps = {
      rowCount: 8,
      predictorCount: 1,
      regimeCount: 2,
      common: [false, true],
      predictors: 'predictors',
      response: 'response',
      regimes: 'regimes',
      robust: {
        type: 'hac' as const,
        kernelWeights: {
          offsets: 'kernelOffsets',
          neighbors: 'kernelNeighbors',
          weights: 'kernelWeights',
          status: {overflow: 'kernelOverflow'}
        }
      },
      output: {
        coefficients: 'coefficients',
        covariance: 'covariance',
        fitted: 'fitted',
        chow: 'chow',
        status: 'status',
        robust: {
          covariance: 'robustCovariance',
          chow: 'robustChow',
          stability: 'robustStability',
          status: 'robustStatus'
        }
      }
    };
    const operation = new GPURestrictedRegimeRegressionOperation(operationProps);
    const cpuOperation = new CPURestrictedRegimeRegressionOperation(operationProps);
    const vector = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    await expect(
      async () =>
        await cpuOperation.execute({
          inputs: {
            predictors: vector(Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3)),
            response: vector(response),
            regimes: vector(regimes),
            kernelOffsets: vector(Uint32Array.from({length: 9}, (_, row) => row)),
            kernelNeighbors: vector(Uint32Array.from({length: 8}, (_, row) => row)),
            kernelWeights: vector(new Float32Array(8).fill(1)),
            kernel_overflow: vector(new Uint32Array(2))
          }
        })
    ).rejects.toThrow(/status.*length/);
    expect(() =>
      constructRegressionOperationContributor(operation, {
        predictors: {format: 'float32', length: 8},
        response: {format: 'float32', length: 8},
        regimes: {format: 'uint32', length: 8},
        kernelOffsets: {format: 'uint32', length: 9},
        kernelNeighbors: {format: 'uint32', length: 8},
        kernelWeights: {format: 'float32', length: 8},
        kernel_overflow: {format: 'uint32', length: 2}
      })
    ).toThrow(/status.*length|status.*one/);
  } finally {
    host.device.destroy();
  }
});

it('R6 rejects one physical coefficients/fitted buffer across read/write prediction bindings', () => {
  const host = createHost();
  try {
    const storage = host.floatView(128);
    const coefficients = host.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 3,
      byteOffset: 0
    });
    const fitted = host.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 8,
      byteOffset: 256
    });
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...host.props,
          output: {...host.props.output, coefficients, fitted}
        })
    ).toThrow(/physical buffer|read-only/);
  } finally {
    host.device.destroy();
  }
});

it('R6 accepts disjoint fitted and robust chow prefixes written by different dispatches', () => {
  const host = createHost();
  try {
    const storage = host.floatView(128);
    const fitted = host.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 8,
      byteOffset: 0
    });
    const chow = host.graph.createDataView(storage.buffer, {
      format: 'float32',
      length: 3,
      byteOffset: 156
    });
    expect(
      () =>
        new GPURestrictedRegimeRegression({
          ...host.props,
          output: {...host.props.output, fitted, robust: {...host.props.output.robust, chow}}
        })
    ).not.toThrow();
  } finally {
    host.device.destroy();
  }
});

it('R5 retains the identified joint HC0 test after an exactly representable predictor shift', () => {
  for (const shift of [0, 1e6]) {
    const result = computeRestrictedRegimeRegressionOnCPU(
      {
        predictors: Float32Array.from([0, 0, 1, 1, 0, 0, 1, 1], value => shift + value / 16),
        regimes,
        response
      },
      {predictorCount: 1, regimeCount: 2, common: [false, false], robust: {type: 'white'}}
    );
    expect(result.robust!.status[0]).toBe(0);
    expect(result.robust!.chow[0]).toBeCloseTo(2.889812889812891, 5);
    expect(result.robust!.chow[1]).toBe(2);
    expect(result.robust!.chow[2]).toBeCloseTo(0.2357681328338333, 6);
  }
});

it('R4 treats requiredCount above capacity as incomplete even without overflow', () => {
  for (const excess of [0, 2]) {
    const result = computeRestrictedRegimeRegressionOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        regimes: Uint32Array.from(fixture.regimes),
        response: Float32Array.from(fixture.response),
        mask: Uint32Array.from(fixture.mask)
      },
      {
        predictorCount: 2,
        regimeCount: 3,
        common: [false, true, false],
        robust: {
          type: 'hac',
          kernelWeights: {
            offsets: Uint32Array.from(fixture.offsets),
            neighbors: Uint32Array.from(fixture.neighbors),
            weights: Float32Array.from(fixture.weights),
            status: {requiredCount: Uint32Array.of(fixture.neighbors.length + excess)}
          }
        }
      }
    );
    expect(result.robust!.status[0]).toBe(excess ? 2 : 0);
    if (excess) expectNaNCovariance(result.robust!.covariance, 7);
    else expect(result.robust!.chow[0]).toBeCloseTo(fixture.expected.hac.chow[0], 5);
  }
});

it('R3 retains restricted HC0 inference when only the unrestricted fit is singular', () => {
  const result = computeRestrictedRegimeRegressionOnCPU(
    {predictors: Float32Array.of(0, 0, 0, 1, 0, 2, 0, 3), regimes, response},
    {predictorCount: 1, regimeCount: 2, common: [false, true], robust: {type: 'white'}}
  );
  expect(result.status[0]).toBe(3);
  expect(result.robust!.status[0]).toBe(0);
  expect(Array.from(result.coefficients)).toEqual([4.25, 3.5, 3]);
  for (const [slot, expected] of [
    [0, 3.296875],
    [4, 4.375],
    [8, 1.25]
  ])
    expect(result.robust!.covariance[slot]).toBeCloseTo(expected, 6);
  expect(result.robust!.chow[0]).toBeCloseTo(0.073319756, 7);
  expect(result.robust!.chow[2]).toBeCloseTo(0.786563159, 7);
});

function identityKernel(capacity = 8) {
  return {
    offsets: Uint32Array.from({length: 9}, (_, row) => row),
    neighbors: Uint32Array.from({length: capacity}, (_, row) => row),
    weights: new Float32Array(capacity).fill(1),
    status: {requiredCount: Uint32Array.of(8), complete: Uint32Array.of(1)}
  };
}

it('W5-41 compares requirements with retained edges and honors complete zero on one allocation', () => {
  const kernel = identityKernel(10);
  for (const [required, complete, expected] of [
    [8, 1, 0],
    [10, 1, 2],
    [8, 1, 0],
    [8, 0, 2]
  ]) {
    kernel.status.requiredCount[0] = required;
    kernel.status.complete[0] = complete;
    const result = computeRestrictedRegimeRegressionOnCPU(
      {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
      {
        predictorCount: 1,
        regimeCount: 2,
        common: [false, true],
        robust: {type: 'hac', kernelWeights: kernel}
      }
    );
    expect(result.robust!.status[0]).toBe(expected);
    // Independent dense HC0 oracle in the review: identity HAC is exactly HC0.
    if (!expected) expect(result.robust!.chow[0]).toBeCloseTo(6.2981105, 5);
    else expectNaNCovariance(result.robust!.covariance, 3);
  }
});

it('W5-41 rejects CSR beyond its reported count and malformed accessible offsets', () => {
  for (const malformed of ['count', 'offsets']) {
    const kernel = {
      ...identityKernel(10),
      status: {
        requiredCount: Uint32Array.of(8),
        count: Uint32Array.of(malformed === 'count' ? 7 : 8)
      }
    };
    if (malformed === 'offsets') kernel.offsets[4] = 9;
    const result = computeRestrictedRegimeRegressionOnCPU(
      {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
      {
        predictorCount: 1,
        regimeCount: 2,
        common: [false, true],
        robust: {type: 'hac', kernelWeights: kernel}
      }
    );
    expect(result.robust!.status[0]).toBe(2);
    expectNaNCovariance(result.robust!.covariance, 3);
  }
});

it('W5-42 rejects the rank-one two-block HAC joint contrast', () => {
  const groups = [1, 0, 0, 0, 0, 1, 0, 0];
  const neighbors = groups.flatMap(group =>
    groups.flatMap((other, row) => (other === group ? [row] : []))
  );
  let retained = 0;
  const offsets = Uint32Array.from([
    ...groups.map(group => {
      const begin = retained;
      retained += groups.filter(other => other === group).length;
      return begin;
    }),
    retained
  ]);
  const result = computeRestrictedRegimeRegressionOnCPU(
    {
      predictors: Float32Array.of(6, 3, 6, 2, 1, 5, 1, 1),
      regimes,
      response: Float32Array.of(0, 8, 6, 4, 8, 8, 7, 1)
    },
    {
      predictorCount: 1,
      regimeCount: 2,
      common: [false, false],
      robust: {
        type: 'hac',
        kernelWeights: {
          offsets,
          neighbors: Uint32Array.from(neighbors),
          weights: new Float32Array(neighbors.length).fill(1)
        }
      }
    }
  );
  // The two block score sums are opposites (X'e=0), so their outer products have rank one.
  expect(result.robust!.status[0]).toBe(3);
  expectNaNCovariance(result.robust!.covariance, 4);
  expect(Array.from(result.robust!.chow)).toEqual([NaN, 0, NaN]);
});

it('W5-43 distinguishes exact-fit roundoff from a small representable residual', () => {
  const predictors = Float32Array.of(3, 2, 6, 5, 1, 2, 6, 6);
  for (const perturbation of [0, 2 ** -10]) {
    const observed = Float32Array.from(
      predictors,
      (value, row) => 3 * value + 4 * regimes[row] + (row === 0 ? perturbation : 0)
    );
    for (const robust of [
      {type: 'white' as const},
      {type: 'hac' as const, kernelWeights: identityKernel()}
    ]) {
      const result = computeRestrictedRegimeRegressionOnCPU(
        {predictors, regimes, response: observed},
        {predictorCount: 1, regimeCount: 2, common: [false, false], robust}
      );
      expect(result.robust!.status[0]).toBe(perturbation ? 0 : 3);
      if (!perturbation) expectNaNCovariance(result.robust!.covariance, 4);
      else {
        // Exact rational normal equations and HC0: Wald=22031625132/25 (oracle.py).
        expect(Number.isFinite(result.robust!.chow[0])).toBe(true);
        expect(Array.from(result.robust!.chow.subarray(1))).toEqual([2, 0]);
        expect(result.robust!.chow.length).toBe(3);
        expect(Math.abs(result.robust!.chow[0] / 881265005.28 - 1)).toBeLessThan(1e-6);
      }
    }
  }
});

it('W5-44 accepts the identified 1024-row half-identity HAC contrast', () => {
  const rowCount = 1024;
  const result = computeRestrictedRegimeRegressionOnCPU(
    {
      predictors: Float32Array.from({length: rowCount}, (_, row) => (row % 8 < 4 ? -1 : 1)),
      regimes: Uint32Array.from({length: rowCount}, (_, row) => row % 2),
      response: Float32Array.from(
        {length: rowCount},
        (_, row) => (row % 4 < 2 ? 1 : -1) + (row % 2) / 16
      )
    },
    {
      predictorCount: 1,
      regimeCount: 2,
      common: [false, true],
      robust: {
        type: 'hac',
        kernelWeights: {
          offsets: Uint32Array.from({length: rowCount + 1}, (_, row) => row * rowCount),
          neighbors: Uint32Array.from({length: rowCount ** 2}, (_, slot) => slot % rowCount),
          weights: Float32Array.from({length: rowCount ** 2}, (_, slot) =>
            Math.floor(slot / rowCount) === slot % rowCount ? 1 : 0.5
          )
        }
      }
    }
  );
  // X'e=0 annihilates 11'; K=(I+11')/2 leaves exactly half of HC0.
  expect(result.robust!.status[0]).toBe(0);
  expect(
    result.robust!.covariance[0] + result.robust!.covariance[4] - 2 * result.robust!.covariance[1]
  ).toBeCloseTo(0.001953125, 9);
  expect(result.robust!.chow[0]).toBeCloseTo(2, 6);
  expect(result.robust!.chow[2]).toBeCloseTo(0.15729921, 7);
});

it('W5-45 preserves classical insufficient-row precedence when restricted fitting fails', () => {
  for (const robust of [undefined, {type: 'white' as const}]) {
    const result = computeRestrictedRegimeRegressionOnCPU(
      {
        predictors: new Float32Array(4),
        regimes: Uint32Array.of(0, 1, 0, 1),
        response: Float32Array.of(0, 1, 4, 9)
      },
      {predictorCount: 1, regimeCount: 2, common: [false, true], robust}
    );
    expect(result.status[0]).toBe(1);
    if (robust) expect(result.robust!.status[0]).toBe(1);
  }
});

it('W5-46 chooses a conditioned joint contrast basis independently of row order', () => {
  const predictors = [20000000, 0, 0, 1, 1, 2, 2, 3];
  for (const order of [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [2, 1, 0, 3, 4, 5, 6, 7]
  ]) {
    const result = computeRestrictedRegimeRegressionOnCPU(
      {
        predictors: Float32Array.from(order, row => predictors[row]),
        regimes: Uint32Array.from(order, row => regimes[row]),
        response: Float32Array.from(order, row => response[row])
      },
      {predictorCount: 1, regimeCount: 2, common: [false, false], robust: {type: 'white'}}
    );
    // Independent NumPy SVD oracle from review R5 (same complete rows, two permutations).
    expect(result.robust!.status[0]).toBe(0);
    expect(result.robust!.chow[0]).toBeCloseTo(8.34675964, 5);
    expect(result.robust!.chow[2]).toBeCloseTo(0.01540012, 7);
  }
});

it('W5-47 captures a kernel status accessor once before every validation and binding', () => {
  const host = createHost();
  try {
    const validated = host.uintView(1);
    const substituted = host.uintView(2);
    let reads = 0;
    const contributor = new GPURestrictedRegimeRegression({
      ...host.props,
      robust: {
        type: 'hac',
        kernelWeights: {
          offsets: host.uintView(9),
          neighbors: host.uintView(8),
          weights: host.floatView(8),
          status: {
            get overflow() {
              return ++reads === 1 ? validated : substituted;
            }
          }
        }
      }
    });
    expect(reads).toBe(1);
    expect(
      contributor.props.robust?.type === 'hac' &&
        contributor.props.robust.kernelWeights.status?.overflow
    ).toBe(validated);
    contributor.getCommandNodes(host.graph);
    expect(reads).toBe(1);
  } finally {
    host.device.destroy();
  }
});

it('B2 retains exact-rational small-residual inference in forward and reverse order', () => {
  const predictors = [1024, 0, 1, 1, 2, 2, 3, 3];
  for (const order of [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [7, 6, 5, 4, 3, 2, 1, 0]
  ]) {
    const result = computeRestrictedRegimeRegressionOnCPU(
      {
        predictors: Float32Array.from(order, row => predictors[row]),
        regimes: Uint32Array.from(order, row => row % 2),
        response: Float32Array.from(
          order,
          row => 3 * predictors[row] + 4 * (row % 2) + (row === 4 ? 1 / 1024 : 0)
        )
      },
      {predictorCount: 1, regimeCount: 2, common: [false, false], robust: {type: 'white'}}
    );
    // Exact rational normal equations and HC0: positive determinant, Wald=7.88566816097316e13.
    expect(result.robust!.status[0]).toBe(0);
    expect(result.robust!.chow.length).toBe(3);
    expect(Number.isFinite(result.robust!.chow[0])).toBe(true);
    expect(Math.abs(result.robust!.chow[0] / 7.88566816097316e13 - 1)).toBeLessThan(1e-5);
    expect(Array.from(result.robust!.chow.subarray(1))).toEqual([2, 0]);
  }
});

it('B4 distinguishes resolved fallback counts from unresolved loss and recovers', () => {
  const kernel = {
    ...identityKernel(),
    status: {uncertainCount: Uint32Array.of(1), unresolvedCount: Uint32Array.of(0)}
  };
  for (const unresolved of [0, 1, 0]) {
    kernel.status.unresolvedCount[0] = unresolved;
    const result = computeRestrictedRegimeRegressionOnCPU(
      {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
      {
        predictorCount: 1,
        regimeCount: 2,
        common: [false, true],
        robust: {type: 'hac', kernelWeights: kernel}
      }
    );
    expect(result.robust!.status[0]).toBe(unresolved ? 2 : 0);
    if (!unresolved) expect(result.robust!.chow[0]).toBeCloseTo(6.2981105, 5);
  }
});

it('S1 rejects both retained-count mismatch directions including count beyond capacity', () => {
  for (const count of [7, 9, 17]) {
    const kernel = {...identityKernel(10), status: {count: Uint32Array.of(count)}};
    const result = computeRestrictedRegimeRegressionOnCPU(
      {predictors: Float32Array.of(0, 0, 1, 1, 2, 2, 3, 3), regimes, response},
      {
        predictorCount: 1,
        regimeCount: 2,
        common: [false, true],
        robust: {type: 'hac', kernelWeights: kernel}
      }
    );
    expect(result.robust!.status[0]).toBe(2);
  }
});

it('B5 captures prototype completeness and information getters once in the contributor', () => {
  const host = createHost();
  try {
    let reads = 0;
    const complete = host.uintView(1);
    class Status {
      get complete() {
        reads++;
        return complete;
      }
    }
    class Information {
      get rowCount() {
        return 8;
      }
      get transform() {
        return 'kernel' as const;
      }
      get symmetric() {
        return true;
      }
      get hasSelfWeights() {
        return true;
      }
    }
    const contributor = new GPURestrictedRegimeRegression({
      ...host.props,
      robust: {
        type: 'hac',
        kernelWeights: {
          offsets: host.uintView(9),
          neighbors: host.uintView(8),
          weights: host.floatView(8),
          status: new Status(),
          information: new Information()
        }
      }
    });
    expect(reads).toBe(1);
    expect(
      contributor.props.robust?.type === 'hac' &&
        contributor.props.robust.kernelWeights.status?.complete
    ).toBe(complete);
    expect(
      contributor.getCommandNodes(host.graph).some(node => node.id.includes('robust-kernel-status'))
    ).toBe(true);
    expect(reads).toBe(1);
  } finally {
    host.device.destroy();
  }
});

it('B5 preserves prototype kernel graph names through the CPU operation snapshot', () => {
  let reads = 0;
  class Status {
    get complete() {
      reads++;
      return 'kernelComplete';
    }
  }
  class Kernel {
    get offsets() {
      return 'kernelOffsets';
    }
    get neighbors() {
      return 'kernelNeighbors';
    }
    get weights() {
      return 'kernelWeights';
    }
    get status() {
      return new Status();
    }
    get information() {
      return new Information();
    }
  }
  class Information {
    get rowCount() {
      return 8;
    }
    get transform() {
      return 'kernel' as const;
    }
    get symmetric() {
      return true;
    }
    get hasSelfWeights() {
      return true;
    }
  }
  const operation = new CPURestrictedRegimeRegressionOperation({
    rowCount: 8,
    predictorCount: 1,
    regimeCount: 2,
    common: [false, true],
    predictors: 'predictors',
    response: 'response',
    regimes: 'regimes',
    robust: {
      type: 'hac',
      kernelWeights: new Kernel()
    },
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      fitted: 'fitted',
      chow: 'chow',
      status: 'status',
      robust: {
        covariance: 'robustCovariance',
        chow: 'robustChow',
        stability: 'robustStability',
        status: 'robustStatus'
      }
    }
  });
  expect(operation.inputs).toHaveProperty('kernel_complete');
  expect(reads).toBe(1);
});

function expectNaNCovariance(values: ArrayLike<number>, coefficientCount: number) {
  expect(values.length).toBe(coefficientCount * coefficientCount);
  expect(Array.from(values).every(Number.isNaN)).toBe(true);
}

it('S2 rejects empty, truncated and overlong NaN covariance fixtures', () => {
  for (const length of [0, 8, 10])
    expect(() => expectNaNCovariance(new Float32Array(length).fill(NaN), 3)).toThrow();
  expectNaNCovariance(new Float32Array(9).fill(NaN), 3);
});

it('B1 B3 projected robust kernels parse and obey binding and array-memory limits', () => {
  const host = createHost();
  try {
    for (const regimeCount of [2, 8])
      for (const robust of [
        {type: 'white' as const},
        {
          type: 'hac' as const,
          kernelWeights: {
            offsets: host.uintView(9),
            neighbors: host.uintView(10),
            weights: host.floatView(10),
            status: {unresolvedCount: host.uintView(1)}
          }
        }
      ]) {
        const contributor = new GPURestrictedRegimeRegression({
          ...host.props,
          id: `review-projected-${regimeCount}-${robust.type}`,
          regimeCount,
          common: [false, false],
          output: {
            ...host.props.output,
            coefficients: host.floatView(2 * regimeCount),
            covariance: host.floatView(4 * regimeCount ** 2),
            robust: {...host.props.output.robust, covariance: host.floatView(4 * regimeCount ** 2)}
          },
          robust
        });
        for (const kernel of getGPUCommandNodeKernelSources(
          contributor.getCommandNodes(host.graph)
        )) {
          expect(new WgslReflect(kernel.source).storage.length, kernel.id).toBeLessThanOrEqual(8);
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        }
      }
  } finally {
    host.device.destroy();
  }
});

it('S3 bounds predictor copying independently of the number of regime contrasts', () => {
  const rowCount = 32;
  const values = Float32Array.from({length: rowCount}, (_, row) => Math.floor(row / 4));
  let reads = 0;
  const predictors = new Proxy(values, {
    get(target, property) {
      if (typeof property === 'string' && /^\d+$/.test(property)) reads++;
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    }
  });
  const result = computeRestrictedRegimeRegressionOnCPU(
    {
      predictors,
      regimes: Uint32Array.from({length: rowCount}, (_, row) => row % 4),
      response: Float32Array.from(
        {length: rowCount},
        (_, row) => 3 * Math.floor(row / 4) + (row % 4) + (Math.floor(row / 8) % 2 ? 1 : -1)
      )
    },
    {predictorCount: 1, regimeCount: 4, common: [false, false], robust: {type: 'white'}}
  );
  expect(result.robust!.status[0]).toBe(0);
  // Conditioning/capture may read each predictor twice; three joint regime contrasts must not
  // copy/sort all predictors afresh. Design subarray access is separate from these origin reads.
  expect(reads).toBeLessThanOrEqual(2 * rowCount);
});

it('B5 preserves prototype kernel graph names through the GPU operation snapshot', () => {
  let reads = 0;
  class Status {
    get complete() {
      reads++;
      return 'kernelComplete';
    }
  }
  class Kernel {
    get offsets() {
      return 'kernelOffsets';
    }
    get neighbors() {
      return 'kernelNeighbors';
    }
    get weights() {
      return 'kernelWeights';
    }
    get status() {
      return new Status();
    }
    get information() {
      return new Information();
    }
  }
  class Information {
    get rowCount() {
      return 8;
    }
    get transform() {
      return 'kernel' as const;
    }
    get symmetric() {
      return true;
    }
    get hasSelfWeights() {
      return true;
    }
  }
  const operation = new GPURestrictedRegimeRegressionOperation({
    rowCount: 8,
    predictorCount: 1,
    regimeCount: 2,
    common: [false, true],
    predictors: 'predictors',
    response: 'response',
    regimes: 'regimes',
    robust: {
      type: 'hac',
      kernelWeights: new Kernel()
    },
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      fitted: 'fitted',
      chow: 'chow',
      status: 'status',
      robust: {
        covariance: 'robustCovariance',
        chow: 'robustChow',
        stability: 'robustStability',
        status: 'robustStatus'
      }
    }
  });
  expect(operation.inputs).toHaveProperty('kernel_complete');
  expect(reads).toBe(1);
});
