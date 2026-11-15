// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  getGPUGTWRParameterValues,
  getGPUMGTWRParameterValues,
  getGPUGTWRBandwidthSearchParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regression';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createGTWRBandwidthSearchNodes,
  createMGTWRComponentEvaluationNodes
} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-bandwidth-search';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/gtwr-bandwidth-search.json';

// Nested profile searches deliberately compile thousands of bounded command nodes.
vi.setConfig({testTimeout: 120_000});

type Configuration = (typeof fixture.cases)[number];
async function runGPU(
  configuration: Configuration,
  options: {
    search?: boolean;
    maximumIterations?: number;
    multiscale?: boolean;
    mask?: number[];
    bounds?: number[];
    fixed?: number[];
    response?: number[];
    predictors?: number[];
    times?: number[];
    rawBounds?: number[];
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    values: number[],
    format: Format,
    components = 1
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? new Uint32Array(values) : new Float32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      values.length / components
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length),
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  const coefficients = output(fixture.rows * 2, 'float32');
  const fitted = output(fixture.rows, 'float32');
  const residuals = output(fixture.rows, 'float32');
  const localStatus = output(fixture.rows, 'uint32');
  const summary = output(12, 'float32');
  const hat = output(fixture.rows, 'float32');
  const selected = output(options.multiscale ? 6 : 2, 'float32');
  const iterations = output(1, 'uint32');
  const converged = output(1, 'uint32');
  const limitReached = output(1, 'uint32');
  const bounds = options.bounds ?? configuration.bounds;
  const fixed = options.fixed ?? configuration.fixed;
  const settings = {bandwidth: fixed[0], timeScale: fixed[1]};
  const common = {
    positions: input(fixture.positions, 'float32x2', 2),
    times: input(options.times ?? fixture.times, 'float32'),
    temporalContext: {
      unit: 'second' as const,
      epoch: 0n,
      encoding: 'float32-relative' as const,
      axis: {kind: 'irregular' as const}
    },
    predictors: input(options.predictors ?? fixture.predictors, 'float32'),
    predictorCount: 1,
    response: input(options.response ?? fixture.response, 'float32'),
    mask: input(options.mask ?? fixture.mask, 'uint32'),
    kernel: configuration.kernel as 'gaussian' | 'bisquare',
    adaptive: configuration.adaptive,
    bandwidthSearch: options.search
      ? {
          maximumIterations: options.maximumIterations ?? 12,
          parameters: input(
            options.rawBounds ??
              Array.from(
                getGPUGTWRBandwidthSearchParameterValues({
                  bandwidthBounds: [bounds[0], bounds[1]],
                  timeScaleBounds: [bounds[2], bounds[3]],
                  absoluteTolerance: 0.01,
                  relativeTolerance: 0.005
                })
              ),
            'float32'
          )
        }
      : undefined
  };
  const searchOutput = options.search
    ? {
        selectedParameters: selected.view,
        status: {
          iterationCount: iterations.view,
          converged: converged.view,
          iterationLimitReached: limitReached.view
        }
      }
    : {};
  if (options.multiscale) {
    graph.add(
      new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
        ...common,
        iterationCount: 1,
        parameters: input(
          Array.from(
            getGPUMGTWRParameterValues({initial: settings, components: [settings, settings]})
          ),
          'float32'
        ),
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          residuals: residuals.view,
          localStatus: localStatus.view,
          scores: output(1, 'float32').view,
          ...searchOutput
        }
      })
    );
  } else {
    graph.add(
      new GPUGeographicallyTemporallyWeightedRegression({
        ...common,
        parameters: input(Array.from(getGPUGTWRParameterValues(settings)), 'float32'),
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          residuals: residuals.view,
          localStatus: localStatus.view,
          hatDiagonal: hat.view,
          summary: summary.view,
          ...searchOutput
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      coefficients: await coefficients.read(),
      fitted: await fitted.read(),
      residuals: await residuals.read(),
      localStatus: await localStatus.read(),
      hat: options.multiscale ? [] : await hat.read(),
      summary: options.multiscale ? [] : await summary.read(),
      selected: options.search ? await selected.read() : [],
      iterations: options.search ? (await iterations.read())[0] : 0,
      converged: options.search ? (await converged.read())[0] : 0,
      limitReached: options.search ? (await limitReached.read())[0] : 0
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function expectIncludedRows(actual: ArrayLike<number>, expected: number[], components = 1) {
  let included = 0;
  fixture.mask.forEach((keep, row) => {
    for (let component = 0; component < components; component++) {
      const value = actual[row * components + component];
      if (!keep) {
        expect(value).toBeNaN();
      } else {
        expect(value).toBeCloseTo(expected[included * components + component], 3);
      }
    }
    if (keep) {
      included++;
    }
  });
}

for (const configuration of fixture.cases) {
  it(`matches NumPy fixed/adaptive GTWR including padding and zero-distance ties: ${configuration.kernel}/${configuration.adaptive}`, async () => {
    const result = await runGPU(configuration);
    expectIncludedRows(result.coefficients, configuration.expected.coefficients, 2);
    expectIncludedRows(result.fitted, configuration.expected.fitted);
    expectIncludedRows(result.hat, configuration.expected.hat);
    expect(result.summary[2]).toBeCloseTo(configuration.expected.aicc, 2);
    fixture.mask.forEach((keep, row) => expect(result.localStatus[row]).toBe(keep ? 0 : 2));
  });
  it(`matches NumPy brute-force grid search AICc: ${configuration.kernel}/${configuration.adaptive}`, async () => {
    const result = await runGPU(configuration, {search: true});
    expect(result.summary[2]).toBeLessThanOrEqual(configuration.grid.aicc + 0.2);
    expect(result.selected[0]).toBeGreaterThanOrEqual(configuration.bounds[0]);
    expect(result.selected[0]).toBeLessThanOrEqual(configuration.bounds[1]);
    expect(result.selected[1]).toBeGreaterThanOrEqual(configuration.bounds[2]);
    expect(result.selected[1]).toBeLessThanOrEqual(configuration.bounds[3]);
    if (configuration.adaptive) {
      expect(result.selected[0]).toBe(Math.round(result.selected[0]));
    }
    // Published selection reproduces the fit independently of the search's scratch/lifetime plan.
    const fixed = await runGPU(configuration, {fixed: Array.from(result.selected)});
    Array.from(result.fitted).forEach((value, row) => {
      if (Number.isNaN(value)) {
        expect(fixed.fitted[row]).toBeNaN();
      } else {
        expect(value).toBeCloseTo(fixed.fitted[row], 4);
      }
    });
  });
}

it('matches NumPy adaptive MGTWR and degenerate-bound profile searches at every component', async () => {
  const configuration = fixture.cases[1];
  for (const search of [false, true]) {
    const result = await runGPU(configuration, {
      multiscale: true,
      search,
      bounds: [22, 22, 1.3, 1.3],
      maximumIterations: 1
    });
    expectIncludedRows(result.coefficients, fixture.multiscale.coefficients, 2);
    expectIncludedRows(result.fitted, fixture.multiscale.fitted);
    expectIncludedRows(result.residuals, fixture.multiscale.residuals);
    if (search) {
      expect(Array.from(result.selected)).toEqual([
        22,
        Math.fround(1.3),
        22,
        Math.fround(1.3),
        22,
        Math.fround(1.3)
      ]);
      expect(result.converged).toBe(1);
      expect(result.limitReached).toBe(0);
      expect(result.iterations).toBe(3);
    }
  }
});

it('publishes the best fit at the iteration limit and handles excluded or singular candidate sets', async () => {
  const configuration = fixture.cases[0];
  const limited = await runGPU(configuration, {search: true, maximumIterations: 1});
  expect(limited.iterations).toBe(1);
  expect(limited.converged).toBe(0);
  expect(limited.limitReached).toBe(1);
  expect(Number.isFinite(limited.summary[2])).toBe(true);
  const empty = await runGPU(configuration, {
    search: true,
    maximumIterations: 1,
    mask: fixture.mask.map(() => 0)
  });
  expect(Array.from(empty.selected).every(Number.isNaN)).toBe(true);
  expect(Array.from(empty.localStatus).every(value => value === 2)).toBe(true);
  const singular = await runGPU(configuration, {
    search: true,
    maximumIterations: 1,
    predictors: fixture.predictors.map(() => 1)
  });
  expect(Array.from(singular.selected).every(Number.isNaN)).toBe(true);
  expect(singular.converged).toBe(0);
  const tooMany = await runGPU(fixture.cases[1], {fixed: [35, 1]});
  fixture.mask.forEach((keep, row) => expect(tooMany.localStatus[row]).toBe(keep ? 1 : 2));
});

it('searches an adaptive bandwidth separately for each MGTWR partial residual against an exhaustive NumPy grid', async () => {
  const result = await runGPU(fixture.cases[1], {
    search: true,
    multiscale: true,
    bounds: [12, 34, 1.3, 1.3]
  });
  expect([result.selected[0], result.selected[2], result.selected[4]]).toEqual(
    fixture.multiscaleSearch.bandwidths
  );
  expectIncludedRows(result.coefficients, fixture.multiscaleSearch.coefficients, 2);
  expectIncludedRows(result.residuals, fixture.multiscaleSearch.residuals);
});

it('keeps the lower time-scale endpoint on score ties and refuses invalid adaptive bounds/counts', async () => {
  const tied = await runGPU(fixture.cases[0], {
    search: true,
    maximumIterations: 1,
    times: fixture.times.map(() => 0),
    bounds: [2, 2, 0, 4]
  });
  expect(tied.selected[0]).toBe(2);
  expect(tied.selected[1]).toBe(0);
  for (const bandwidth of [1, 12.5, 129]) {
    const result = await runGPU(fixture.cases[1], {fixed: [bandwidth, 1]});
    fixture.mask.forEach((keep, row) => expect(result.localStatus[row]).toBe(keep ? 1 : 2));
  }
  const invalid = await runGPU(fixture.cases[1], {
    search: true,
    maximumIterations: 1,
    rawBounds: [12, 129, 0, 4, 0.01, 0.005]
  });
  expect(Array.from(invalid.selected).every(Number.isNaN)).toBe(true);
  expect(invalid.converged).toBe(0);
});

it('retains count 18 when the adaptive golden-section interior candidates coincide', async () => {
  const result = await runGPU(fixture.cases[1], {
    search: true,
    maximumIterations: 12,
    bounds: [15, 19, 1, 1]
  });
  expect(result.selected[0]).toBe(18);
  expect(result.selected[1]).toBe(1);
  expect(result.summary[2]).toBeCloseTo(fixture.roundedCandidates.aicc[3], 2);
  expect(result.converged).toBe(1);
  expect(result.limitReached).toBe(0);
});

it('keeps scaled MGTWR component candidates finite and selects the fixed pair', async () => {
  const result = await runGPU(fixture.cases[1], {
    search: true,
    multiscale: true,
    bounds: [22, 22, 1.3, 1.3],
    maximumIterations: 1,
    predictors: fixture.predictors.map(value => value * 1e14),
    response: fixture.response.map(value => value * 1e12)
  });
  expect(Array.from(result.selected)).toEqual([
    22,
    Math.fround(1.3),
    22,
    Math.fround(1.3),
    22,
    Math.fround(1.3)
  ]);
  expect(result.converged).toBe(1);
  expect(result.limitReached).toBe(0);
  fixture.mask.forEach((keep, row) => {
    expect(result.localStatus[row]).toBe(keep ? 0 : 2);
    if (keep) {
      expect(Number.isFinite(result.fitted[row])).toBe(true);
      expect(Number.isFinite(result.coefficients[row * 2])).toBe(true);
      expect(Number.isFinite(result.coefficients[row * 2 + 1])).toBe(true);
    }
  });
  // Changing units scales beta_0 and y by 1e12 and beta_1 by 1e-2. Compare the
  // independently generated NumPy backfit after undoing those unit conversions.
  expectIncludedRows(
    Array.from(result.fitted, value => value / 1e12),
    fixture.multiscale.fitted
  );
  expectIncludedRows(
    Array.from(result.residuals, value => value / 1e12),
    fixture.multiscale.residuals
  );
  expectIncludedRows(
    Array.from(result.coefficients, (value, index) => (index % 2 ? value * 1e2 : value / 1e12)),
    fixture.multiscale.coefficients,
    2
  );
});

async function runLargeResponseGPU(model: 'component' | 'gtwr') {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2'>(
    values: number[],
    format: Format,
    components = 1
  ) => {
    const buffer = createInputBuffer(device, new Float32Array(values));
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `large-input-${serial++}`,
      buffer,
      format,
      values.length / components
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `large-output-${serial++}`, buffer, format, length),
      buffer
    };
  };
  const selected = output(2, 'float32');
  const summary = output(12, 'float32');
  const converged = output(1, 'uint32');
  const limitReached = output(1, 'uint32');
  const predictors = input(fixture.largeResponse.predictors, 'float32');
  const response = input(fixture.largeResponse.response, 'float32');
  const bounds = input([100, 100, 0, 0, 0.01, 0.005], 'float32');
  const search = {parameters: bounds, maximumIterations: 1};
  const status = {converged: converged.view, iterationLimitReached: limitReached.view};
  if (model === 'component') {
    const points = input(fixture.largeResponse.points, 'float32');
    graph.add(
      createGTWRBandwidthSearchNodes(graph, {
        id: 'large-component-search',
        rows: 8,
        adaptive: false,
        search,
        parameters: selected.view,
        status,
        evaluate: (parameters, gate, id) => {
          const evaluation = createMGTWRComponentEvaluationNodes(graph, {
            id,
            rows: 8,
            predictorCount: 1,
            component: 0,
            points,
            predictors,
            partial: response,
            parameters,
            adaptive: false,
            bisquare: false,
            gate
          });
          return {
            summary: evaluation.summary,
            nodes: [
              ...evaluation.nodes,
              createWGSLKernelNode(graph, {
                id: `${id}-publish-summary`,
                operation: 'LargeResponseOracle',
                invocationCount: 1,
                ...gate.getGate(1),
                bindings: [
                  {name: 'summary', view: evaluation.summary, type: 'f32', access: 'read'},
                  {name: 'output', view: summary.view, type: 'f32', access: 'read_write'}
                ],
                body: `for (var field = 0u; field < 12u; field++) {
  output[outputOffset + field] = summary[summaryOffset + field];
}`
              })
            ]
          };
        }
      })
    );
  } else {
    graph.add(
      new GPUGeographicallyTemporallyWeightedRegression({
        positions: input(fixture.largeResponse.positions, 'float32x2', 2),
        times: input(new Array(8).fill(0), 'float32'),
        temporalContext: {
          unit: 'second',
          epoch: 0n,
          encoding: 'float32-relative',
          axis: {kind: 'irregular'}
        },
        predictors,
        predictorCount: 1,
        response,
        parameters: input([100, 0, 0, 0], 'float32'),
        bandwidthSearch: search,
        output: {
          coefficients: output(16, 'float32').view,
          fitted: output(8, 'float32').view,
          hatDiagonal: output(8, 'float32').view,
          localStatus: output(8, 'uint32').view,
          summary: summary.view,
          selectedParameters: selected.view,
          status
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      selected: await readFloat32(selected.buffer, 2),
      summary: await readFloat32(summary.buffer, 12),
      converged: (await readUint32(converged.buffer, 1))[0],
      limitReached: (await readUint32(limitReached.buffer, 1))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const model of ['component', 'gtwr'] as const) {
  it(`retains [100, 0] for large finite ${model} residual squares without overflowing AICc`, async () => {
    const result = await runLargeResponseGPU(model);
    expect(result.summary[2]).toBeCloseTo(fixture.largeResponse.expected[model].aicc, 3);
    expect(result.summary[4]).toBe(8);
    expect(result.summary[6]).toBe(0);
    expect(Array.from(result.selected)).toEqual([100, 0]);
    expect(result.converged).toBe(1);
    expect(result.limitReached).toBe(0);
  });
}
