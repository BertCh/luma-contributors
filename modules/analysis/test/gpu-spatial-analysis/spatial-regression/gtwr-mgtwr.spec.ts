// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  getGPUGTWRParameterValues,
  getGPUMGTWRParameterValues,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regression';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import oracle from '../../oracles/fixtures/gtwr-mgtwr.json';

type FitInput = {
  rows: number;
  columns: number;
  coords: number[];
  t: number[];
  x: number[];
  y: number[];
  bandwidth: number;
  tau: number;
  kernel: 'gaussian' | 'bisquare';
  mask?: number[];
};

async function runGPU(input: FitInput) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let next = 0;
  const makeInput = (values: number[], format: GPUVectorFormat, components = 1) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? new Uint32Array(values) : new Float32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${next++}`, buffer, format, values.length / components);
  };
  const makeOutput = (length: number, format: GPUVectorFormat) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length),
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  const size = input.columns + 1;
  const coefficients = makeOutput(input.rows * size, 'float32');
  const localStatus = makeOutput(input.rows, 'uint32');
  const fitted = makeOutput(input.rows, 'float32');
  const hatDiagonal = makeOutput(input.rows, 'float32');
  const residuals = makeOutput(input.rows, 'float32');
  const summary = makeOutput(12, 'float32');
  graph.add(
    new GPUGeographicallyTemporallyWeightedRegression({
      positions: makeInput(input.coords, 'float32x2', 2) as never,
      times: makeInput(input.t, 'float32') as never,
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      predictors: makeInput(input.x, 'float32') as never,
      predictorCount: input.columns,
      response: makeInput(input.y, 'float32') as never,
      mask: input.mask && (makeInput(input.mask, 'uint32') as never),
      kernel: input.kernel,
      parameters: makeInput(
        Array.from(getGPUGTWRParameterValues({bandwidth: input.bandwidth, timeScale: input.tau})),
        'float32'
      ) as never,
      output: {
        coefficients: coefficients.view as never,
        localStatus: localStatus.view as never,
        fitted: fitted.view as never,
        hatDiagonal: hatDiagonal.view as never,
        residuals: residuals.view as never,
        summary: summary.view as never
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      coefficients: await coefficients.read(),
      localStatus: await localStatus.read(),
      fitted: await fitted.read(),
      hatDiagonal: await hatDiagonal.read(),
      residuals: await residuals.read(),
      summary: await summary.read()
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

const close = (actual: number, expected: number, tolerance: number, name: string) =>
  expect(Math.abs(actual - expected), `${name}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance * Math.max(1, Math.abs(expected))
  );

for (const fixture of oracle.cases) {
  it(`matches mgtwr GTWR.fit: ${fixture.name} (bandwidth ${fixture.bandwidth}, tau ${fixture.tau})`, async () => {
    const result = await runGPU({...fixture, kernel: fixture.kernel as 'gaussian' | 'bisquare'});
    const size = fixture.columns + 1;
    expect(result.localStatus.every(code => code === 0)).toBe(true);
    for (let row = 0; row < fixture.rows; row++) {
      for (let column = 0; column < size; column++) {
        close(
          result.coefficients[row * size + column],
          fixture.betas[row * size + column],
          2e-3,
          `beta ${row},${column}`
        );
      }
      close(result.fitted[row], fixture.fitted[row], 1e-3, `fitted ${row}`);
      close(result.residuals[row], fixture.residuals[row], 2e-3, `residual ${row}`);
      close(result.hatDiagonal[row], fixture.hat[row], 2e-3, `hat ${row}`);
    }
    // Summary slots follow GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY: RSS, trace, AICc, R2, n, ..., sigma2, llf.
    close(result.summary[0], fixture.rss, 2e-3, 'rss');
    close(result.summary[1], fixture.trace, 2e-3, 'trace');
    // The summary AICc is the GWR (mgwr) formula evaluated on mgtwr's log-likelihood and trace:
    // -2 llf + 2 n (tr + 1) / (n - tr - 2). mgtwr's own aicc differs by a known penalty gap.
    const penaltyGap =
      (2 * fixture.rows * (fixture.trace + 1)) / (fixture.rows - fixture.trace - 2) -
      (2 * (fixture.trace + 1) * (fixture.rows - 1)) / (fixture.rows - fixture.trace - 1);
    close(
      result.summary[2],
      -2 * fixture.llf +
        (2 * fixture.rows * (fixture.trace + 1)) / (fixture.rows - fixture.trace - 2),
      2e-3,
      'aicc'
    );
    close(result.summary[2] - fixture.aicc, penaltyGap, 2e-2, 'aicc gap to mgtwr');
    close(result.summary[3], fixture.r2, 2e-3, 'r2');
    expect(result.summary[4]).toBe(fixture.rows);
    close(result.summary[7], fixture.sigma2, 3e-3, 'sigma2');
    close(result.summary[10], fixture.llf, 2e-3, 'llf');
    expect(result.summary[6]).toBe(0);
    expect(result.summary[11]).toBe(fixture.rows);
  });
}

it('time scale changes the fit: a larger scale differs from a smaller one', async () => {
  const fixture = oracle.cases[0];
  const base = {...fixture, kernel: 'gaussian' as const};
  const small = await runGPU({...base, tau: 0.1});
  const large = await runGPU({...base, tau: 6});
  const difference = small.fitted.reduce(
    (sum, value, row) => sum + Math.abs(value - large.fitted[row]),
    0
  );
  expect(difference).toBeGreaterThan(0.5);
});

it('translating all times leaves the fit unchanged', async () => {
  const fixture = oracle.cases[1];
  const base = {...fixture, kernel: 'bisquare' as const};
  const original = await runGPU(base);
  const shifted = await runGPU({...base, t: base.t.map(value => value + 4)});
  original.fitted.forEach((value, row) => close(shifted.fitted[row], value, 1e-3, `row ${row}`));
});

it('reports singular, excluded and invalid-parameter rows', async () => {
  const fixture = oracle.cases[0];
  const base = {...fixture, kernel: 'gaussian' as const};
  // Duplicate the first predictor into the second: every local design is rank deficient.
  const x = base.x.slice();
  for (let row = 0; row < base.rows; row++) x[row * 2 + 1] = x[row * 2];
  const singular = await runGPU({...base, x});
  expect(
    singular.localStatus.every(
      code => code === GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR
    )
  ).toBe(true);
  expect(singular.coefficients.every(Number.isNaN)).toBe(true);
  expect(singular.summary[6]).toBe(base.rows);
  const mask = base.y.map((_, row) => (row % 5 === 0 ? 0 : 1));
  const masked = await runGPU({...base, mask});
  masked.localStatus.forEach((code, row) =>
    expect(code).toBe(
      mask[row] === 0
        ? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED
        : GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK
    )
  );
  expect(masked.summary[4]).toBe(mask.filter(Boolean).length);
  expect(() => getGPUGTWRParameterValues({bandwidth: 0, timeScale: 1})).toThrow('bandwidth');
  expect(() => getGPUGTWRParameterValues({bandwidth: 1, timeScale: -1})).toThrow('timeScale');
});

type MultiscaleInput = {
  rows: number;
  columns: number;
  kernel: 'gaussian' | 'bisquare';
  initial: number[];
  bandwidths: number[];
  taus: number[];
  iterations: number;
  coords: number[];
  t: number[];
  x: number[];
  y: number[];
  mask?: number[];
};

async function runMultiscaleGPU(input: MultiscaleInput) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let next = 0;
  const makeInput = (values: number[], format: GPUVectorFormat, components = 1) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? new Uint32Array(values) : new Float32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${next++}`, buffer, format, values.length / components);
  };
  const makeOutput = (length: number, format: GPUVectorFormat) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length),
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  const size = input.columns + 1;
  const coefficients = makeOutput(input.rows * size, 'float32');
  const fitted = makeOutput(input.rows, 'float32');
  const residuals = makeOutput(input.rows, 'float32');
  const localStatus = makeOutput(input.rows, 'uint32');
  const scores = makeOutput(input.iterations, 'float32');
  graph.add(
    new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
      positions: makeInput(input.coords, 'float32x2', 2) as never,
      times: makeInput(input.t, 'float32') as never,
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      predictors: makeInput(input.x, 'float32') as never,
      predictorCount: input.columns,
      response: makeInput(input.y, 'float32') as never,
      mask: input.mask && (makeInput(input.mask, 'uint32') as never),
      kernel: input.kernel,
      iterationCount: input.iterations,
      parameters: makeInput(
        Array.from(
          getGPUMGTWRParameterValues({
            initial: {bandwidth: input.initial[0], timeScale: input.initial[1]},
            components: input.bandwidths.map((bandwidth, index) => ({
              bandwidth,
              timeScale: input.taus[index]
            }))
          })
        ),
        'float32'
      ) as never,
      output: {
        coefficients: coefficients.view as never,
        fitted: fitted.view as never,
        residuals: residuals.view as never,
        localStatus: localStatus.view as never,
        scores: scores.view as never
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      coefficients: await coefficients.read(),
      fitted: await fitted.read(),
      residuals: await residuals.read(),
      localStatus: await localStatus.read(),
      scores: await scores.read()
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const fixture of oracle.multiscale) {
  it(`matches mgtwr multi_bws backfitting: ${fixture.name} (${fixture.iterations} iterations)`, async () => {
    const result = await runMultiscaleGPU({
      ...fixture,
      kernel: fixture.kernel as 'gaussian' | 'bisquare'
    });
    expect(result.localStatus.every(code => code === 0)).toBe(true);
    fixture.betas.forEach((value, index) =>
      close(result.coefficients[index], value, 5e-3, `beta ${index}`)
    );
    fixture.residuals.forEach((value, row) =>
      close(result.residuals[row], value, 5e-3, `residual ${row}`)
    );
    // The scores shrink by an order of magnitude per iteration; compare in relative terms at the
    // first iterations and in absolute terms once they approach float32 noise.
    fixture.scores.forEach((value, iteration) => {
      if (value > 1e-4) {
        expect(Math.abs(result.scores[iteration] / value - 1), `score ${iteration}`).toBeLessThan(
          0.02
        );
      } else {
        expect(result.scores[iteration], `score ${iteration}`).toBeLessThan(1e-3);
      }
    });
  });
}

it('multiscale: masked rows are excluded and the scores stay finite', async () => {
  const fixture = oracle.multiscale[0];
  const base = {...fixture, kernel: 'gaussian' as const};
  const mask = base.y.map((_, row) => (row % 7 === 0 ? 0 : 1));
  const result = await runMultiscaleGPU({...base, mask});
  result.localStatus.forEach((code, row) =>
    expect(code).toBe(
      mask[row] === 0
        ? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED
        : GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK
    )
  );
  expect(result.fitted.filter(Number.isNaN).length).toBe(mask.filter(value => value === 0).length);
  expect(result.scores.every(Number.isFinite)).toBe(true);
  expect(() =>
    getGPUMGTWRParameterValues({
      initial: {bandwidth: 1, timeScale: 0},
      components: [{bandwidth: -1, timeScale: 0}]
    })
  ).toThrow('bandwidth');
});

it('multiscale first backfit sweep matches independent estimates before convergence', async () => {
  // Independent NumPy float64 reference, on these exact uploaded values:
  // D=column_stack([ones(8),x]); W_h[i,j]=exp(-.5*((i-j)/h)**2).
  // Initial beta_i=lstsq(sqrt(W_4[i])[:,None]*D,sqrt(W_4[i])*y); XB=D*beta.
  // For j=0,1 in order: r=XB[:,j]+err; beta[:,j]=(W_h@(D[:,j]*r))/(W_h@D[:,j]**2);
  // new=beta[:,j]*D[:,j]; err+=XB[:,j]-new; XB[:,j]=new, h=(.75,3).
  // score=sqrt(sum((XB_new-XB_old)**2)/8/sum(XB_new.sum(axis=1)**2)).
  const input: MultiscaleInput = {
    rows: 8,
    columns: 1,
    kernel: 'gaussian',
    initial: [4, 0],
    bandwidths: [0.75, 3],
    taus: [0, 0],
    iterations: 1,
    coords: Array.from({length: 8}, (_, row) => [row, 0]).flat(),
    t: new Array<number>(8).fill(0),
    x: [-2, -1, 0, 1, 2, -2, 1, 3],
    y: [0, 2, 1, 4, 2, -1, 3, 8]
  };
  const expectedCoefficients = [
    2.3068671958389624, 1.0798844636127893, 2.373646466354355, 1.0985241198635616,
    1.840671349063036, 1.1275134907296904, 1.6723986677152878, 1.1661555910730113,
    0.741984482263636, 1.2119102591421107, 1.2157923124218892, 1.2614249992161484,
    2.0318298142417195, 1.3115474328097951, 3.0070691591458405, 1.3598278733021907
  ];
  const expectedFitted = [
    0.1470982686133837, 1.2751223464907935, 1.840671349063036, 2.838554258788299,
    3.1658050005478575, -1.3070576860104075, 3.3433772470515146, 7.086552779052413
  ];
  const first = await runMultiscaleGPU(input);
  expect(first.localStatus).toHaveLength(8);
  expect(first.coefficients).toHaveLength(16);
  expect(first.fitted).toHaveLength(8);
  expect(first.localStatus.every(code => code === 0)).toBe(true);
  expectedCoefficients.forEach((expected, index) => {
    expect(Number.isFinite(first.coefficients[index])).toBe(true);
    expect(Math.abs(first.coefficients[index] - expected)).toBeLessThan(2e-4);
  });
  expectedFitted.forEach((expected, row) => {
    expect(Number.isFinite(first.fitted[row])).toBe(true);
    expect(Math.abs(first.fitted[row] - expected)).toBeLessThan(2e-4);
  });
  expect(Number.isFinite(first.scores[0])).toBe(true);
  expect(Math.abs(first.scores[0] - 0.07315682360842546)).toBeLessThan(2e-5);
  // iterationCount is a fixed backfit budget. No bandwidthSearch is requested; its optional
  // search convergence status must not be interpreted as convergence of these component fits.
  expect(first.scores).toHaveLength(1);
  const second = await runMultiscaleGPU({...input, iterations: 2});
  expect(second.localStatus).toHaveLength(8);
  expect(second.scores).toHaveLength(2);
  expect(second.localStatus.every(code => code === 0)).toBe(true);
  expect(Number.isFinite(second.scores[1])).toBe(true);
  expect(Math.abs(second.scores[1] - 0.012461856428517642)).toBeLessThan(2e-5);
  expect(Number.isFinite(second.coefficients[14])).toBe(true);
  expect(Math.abs(second.coefficients[14] - 3.234705305106167)).toBeLessThan(2e-4);
  // The last intercept still moves by .227636 after the first sweep.
  expect(Math.abs(second.coefficients[14] - first.coefficients[14])).toBeGreaterThan(0.2);
});
