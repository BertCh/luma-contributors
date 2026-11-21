// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {computeOrdinaryLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {getCPURegressionSingularValueRange} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-regression-linear-algebra';
import {
  getGPUOrdinaryLeastSquaresModelResult,
  getStudentTTwoSidedPValue
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getKernel(rows: number, suffix: string, tileRowCount = 6) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `review2-${serial++}`, format, length);
    const contributor = new GPUOrdinaryLeastSquares({
      predictorCount: 1,
      tileRowCount,
      predictors: view('float32', rows),
      response: view('float32', rows),
      output: {
        coefficients: view('float32', 2),
        standardErrors: view('float32', 2),
        tStatistics: view('float32', 2),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    return getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id.endsWith(suffix))
    )[0];
  } finally {
    device.destroy();
  }
}

for (const scale of [1, 1e100]) {
  it(`R12 Jacobi preserves the small singular value by relative error at scale ${scale}`, () => {
    // The small column has orthogonal component scale*1e-180; corrections to that singular
    // value are O(1e-360) relatively. SciPy independently gives [1,1e-180] at scale one.
    const [minimum, maximum] = getCPURegressionSingularValueRange([
      [scale, scale * 1e-180],
      [0, scale * 1e-180]
    ]);
    expect(Number.isFinite(minimum)).toBe(true);
    expect(Math.abs(minimum / (scale * 1e-180) - 1)).toBeLessThan(1e-12);
    expect(Math.abs(maximum / scale - 1)).toBeLessThan(1e-12);
  });
}

it('R13 decoder captures and validates the same coefficient arrays once', () => {
  const fit = computeOrdinaryLeastSquaresOnCPU(
    {predictors: Float32Array.of(-1, -1, 1, 1), response: Float32Array.of(0, 2, 4, 6)},
    1
  );
  const reads = {coefficients: 0, standardErrors: 0, tStatistics: 0};
  const readback = {
    summary: fit.summary,
    status: fit.status,
    get coefficients() {
      return ++reads.coefficients === 1 ? Float32Array.of(3, 2) : Float32Array.of(3);
    },
    get standardErrors() {
      return ++reads.standardErrors === 1 ? Float32Array.of(1, 1) : Float32Array.of(1);
    },
    get tStatistics() {
      return ++reads.tStatistics === 1 ? Float32Array.of(7, 5) : Float32Array.of(7);
    }
  };
  const result = getGPUOrdinaryLeastSquaresModelResult(readback, {predictorCount: 1});
  expect(result.coefficients[1].estimate).toBe(2);
  expect(result.coefficients[1].standardError).toBe(1);
  expect(result.coefficients[1].statistic).toBe(5);
  // With four rows and two coefficients df=2; the closed-form two-sided t tail is
  // 1-|t|/sqrt(t²+2), independent of the decoder or the production beta evaluator.
  expect(result.coefficients[1].pValue).toBeCloseTo(1 - 5 / Math.sqrt(27), 13);
  expect(reads).toEqual({coefficients: 1, standardErrors: 1, tStatistics: 1});
});

it('R14 large-df Student t retains accurate representable tails', () => {
  // Independent scipy.stats.t.sf values in audit-ols-review2.oracle.py.
  for (const [statistic, freedom, expected] of [
    [12, 1e20, 3.552964224155406e-33],
    [2, 1e15, 0.045500263896358695],
    [12, 1e10, 3.552966091509718e-33],
    [35, 1e10, 2.2499059557285326e-268],
    [0.1, 1e10, 0.9203443254479469]
  ]) {
    const actual = getStudentTTwoSidedPValue(statistic, freedom);
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBeGreaterThan(0);
    expect(actual).toBeLessThanOrEqual(1);
    expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-12);
  }
});

function getSlot(source: string, name: string) {
  const match = source.match(new RegExp(`const ${name}: u32 = ([0-9]+)u;`));
  if (!match) throw new Error(`Missing ${name}`);
  return Number(match[1]);
}

function execute(
  kernel: ReturnType<typeof getKernel>,
  values: Record<string, ArrayBufferView>,
  workgroups: number
) {
  const bindings: Record<number, ArrayBufferView> = {};
  for (const match of kernel.source.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var<storage,\s*\w+>\s*(\w+)\s*:/g
  )) {
    const value = values[match[2]];
    if (!value) throw new Error(`Missing binding ${match[2]}`);
    bindings[Number(match[1])] = value;
  }
  // Interpreter compatibility only: phony assignments and one-component workgroup attributes.
  const source = kernel.source
    .replace(/^\s*_\s*=[^;]*;/gm, '')
    .replace(/@workgroup_size\(\s*(\d+)u?\s*\)/g, '@workgroup_size($1,1,1)');
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    [workgroups, 1, 1],
    {0: bindings},
    {constants: kernel.constants}
  );
}

it('R10 actual normalized auxiliary WGSL retains nearly constant squared-magnitude differences', () => {
  const difference = 2 ** -23;
  const first = 1.375 + 3 * difference;
  const residuals = Float32Array.from(
    Array.from({length: 64}, () => [
      -first,
      first,
      -(first + difference),
      first + difference,
      -(first + 2 * difference),
      first + 2 * difference
    ]).flat()
  );
  const kernel = getKernel(384, '-standardized-residual-tiles');
  const workspace = new Uint32Array(64);
  new Float32Array(workspace.buffer)[getSlot(kernel.source, 'W_RESIDUAL_SCALE')] =
    first + 2 * difference;
  const auxiliary = new Float32Array(384);
  execute(
    kernel,
    {
      residualIn: residuals,
      ws: workspace,
      partial: new Float32Array(64 * 3),
      squaredOut: auxiliary
    },
    1
  );
  // Exact rational e²/scale²-1, generated by audit-ols-review2.oracle.py. The centered
  // levels have f32 relative resolution; storing e²/scale² near one discarded this resolution.
  const expected = [
    -3.467904801158115e-7, -3.467904801158115e-7, -1.733952475743844e-7, -1.733952475743844e-7, 0, 0
  ];
  auxiliary.forEach((value, row) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - auxiliary[5] - expected[row % 6])).toBeLessThan(6e-14);
  });
  const fit = computeOrdinaryLeastSquaresOnCPU(
    {
      predictors: Float32Array.from(Array.from({length: 64}, () => [1, 1, -2, -2, 1, 1]).flat()),
      response: residuals
    },
    1
  );
  expect(Array.from(fit.coefficients)).toEqual([0, 0]);
  expect(Math.abs(fit.summary[9] / 2.4052735761050423e-13 - 1)).toBeLessThan(0.03);
  expect(fit.summary[10]).toBeCloseTo(0.999999608688779, 7);
});

for (const mode of ['varying', 'constant', 'exact'] as const) {
  it(`R11 actual BP finish uses normalized evidence instead of rounded raw RSS: ${mode}`, () => {
    const kernel = getKernel(16, '-bp-finish');
    const workspace = new Uint32Array(64);
    const floats = new Float32Array(workspace.buffer);
    workspace[getSlot(kernel.source, 'W_COUNT')] = 16;
    // Independent moments for x=[-1,-1,1,1] repeated four times and centered normalized
    // auxiliary response [-.75,-.75,0,0]: X'X=16, X'q=6, q'q=2.25, slope=3/8.
    const moments = getSlot(kernel.source, 'W_MOMENTS');
    floats[moments] = 16;
    if (mode === 'varying') {
      floats[moments + 1] = 6;
      floats[moments + 2] = 2.25;
      floats[getSlot(kernel.source, 'W_SLOPES')] = 0.375;
    }
    const mainWorkspace = new Uint32Array(64);
    const mainFloats = new Float32Array(mainWorkspace.buffer);
    // At scale 2^-76, all individual raw squares round to zero, but normalized SS=10.
    mainFloats[getSlot(kernel.source, 'W_STANDARDIZED_SUMS')] =
      mode === 'varying' ? 10 : mode === 'constant' ? 16 : 0;
    const summary = new Float32Array(22).fill(NaN);
    execute(kernel, {ws: workspace, mainWs: mainWorkspace, summaryOut: summary}, 1);
    if (mode === 'exact') {
      expect(summary[16]).toBeNaN();
      expect(summary[17]).toBeNaN();
    } else {
      const expected = mode === 'varying' ? [2.88, 0.08968602177036485] : [0, 1];
      expected.forEach((value, index) => {
        expect(Number.isFinite(summary[16 + index])).toBe(true);
        expect(Math.abs(summary[16 + index] - value)).toBeLessThan(2e-6);
      });
    }
    const scale = 2 ** -76;
    const predictors = Float32Array.from(Array.from({length: 4}, () => [-1, -1, 1, 1]).flat());
    const response =
      mode === 'exact'
        ? predictors
        : Float32Array.from(
            Array.from({length: 4}, () =>
              mode === 'varying'
                ? [-scale, scale, -2 * scale, 2 * scale]
                : [-scale, scale, -scale, scale]
            ).flat()
          );
    const host = computeOrdinaryLeastSquaresOnCPU({predictors, response}, 1);
    expect(host.status[0]).toBe(0);
    if (mode === 'exact') {
      expect(host.summary[16]).toBeNaN();
      expect(host.summary[17]).toBeNaN();
    } else {
      const expected = mode === 'varying' ? [2.88, 0.08968602177036485] : [0, 1];
      expected.forEach((value, index) => expect(host.summary[16 + index]).toBeCloseTo(value, 6));
    }
  });
}
