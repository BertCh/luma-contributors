// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import type {GPUOrdinaryLeastSquaresProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {
  getGPUOrdinaryLeastSquaresParameterValues,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {
  getChiSquareSurvival,
  getLogGammaOfHalfInteger
} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createOrdinaryLeastSquaresScene,
  fitOrdinaryLeastSquaresOnCPU
} from './ordinary-least-squares-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUOrdinaryLeastSquaresProps> = {}
): GPUOrdinaryLeastSquaresProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    predictors: view('float32', 40),
    response: view('float32', 20),
    predictorCount: 2,
    output: {
      coefficients: view('float32', 3),
      standardErrors: view('float32', 3),
      tStatistics: view('float32', 3),
      summary: view('float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
      status: {code: view('uint32', 1)}
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUOrdinaryLeastSquaresProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUOrdinaryLeastSquares(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('GPUOrdinaryLeastSquares validates its inputs', () => {
  const view = <Format extends 'uint32' | 'float32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(() => ({predictorCount: 0}), /predictorCount/);
  expectThrows(() => ({predictorCount: 16}), /predictorCount/);
  expectThrows(() => ({predictorCount: 1.5}), /predictorCount/);
  expectThrows(graph => ({predictors: view(graph, 'float32', 39)}), /predictors length/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 19)}), /mask length/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 0)}), /parameters must hold/);
  expectThrows(() => ({tileRowCount: 0}), /tileRowCount/);
  expectThrows(
    graph => ({
      output: {
        coefficients: view(graph, 'float32', 2),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
        status: {code: view(graph, 'uint32', 1)}
      }
    }),
    /coefficients must hold/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: view(graph, 'float32', 3),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH - 1),
        status: {code: view(graph, 'uint32', 1)}
      }
    }),
    /summary must hold/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: view(graph, 'float32', 3),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
        status: {code: view(graph, 'uint32', 1)},
        residuals: view(graph, 'float32', 19)
      }
    }),
    /residuals must hold/
  );
  expectThrows(graph => {
    const shared = view(graph, 'float32', 20);
    return {
      response: shared,
      output: {
        coefficients: view(graph, 'float32', 3),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
        status: {code: view(graph, 'uint32', 1)},
        residuals: shared
      }
    };
  }, /must not share buffers/);
});

it('GPUOrdinaryLeastSquares emits a deterministic node chain with stable IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUOrdinaryLeastSquares(createProps(graph, {id: 'ols', tileRowCount: 8}));
  expect(contributor.tileCount).toBe(3);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'ols-response-magnitudes',
    'ols-response-maximum-level-0',
    'ols-response-maximum-finalize',
    'ols-response-normalize',
    'ols-means-origins',
    'ols-means-tiles',
    'ols-means-merge',
    'ols-moments-tiles',
    'ols-moments-merge',
    'ols-solve',
    'ols-residuals',
    'ols-residual-tiles',
    'ols-residual-merge',
    'ols-standardized-residual-tiles',
    'ols-standardized-residual-merge',
    'ols-finish',
    'ols-scaled-likelihood',
    'ols-bp-means-origins',
    'ols-bp-means-tiles',
    'ols-bp-means-merge',
    'ols-bp-moments-tiles',
    'ols-bp-moments-merge',
    'ols-bp-solve',
    'ols-bp-finish',
    'ols-response-restore-report'
  ]);
  device.destroy();
});

it('default tile size caps the tile count at 4096', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = 3_000_000;
  const contributor = new GPUOrdinaryLeastSquares({
    predictors: createTransientView(graph, 'p', 'float32', rows),
    response: createTransientView(graph, 'r', 'float32', rows),
    predictorCount: 1,
    output: {
      coefficients: createTransientView(graph, 'c', 'float32', 2),
      standardErrors: createTransientView(graph, 's', 'float32', 2),
      tStatistics: createTransientView(graph, 't', 'float32', 2),
      summary: createTransientView(
        graph,
        'm',
        'float32',
        GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
      ),
      status: {code: createTransientView(graph, 'u', 'uint32', 1)}
    }
  });
  expect(contributor.tileCount).toBeLessThanOrEqual(4096);
  expect(contributor.tileRowCount).toBe(Math.ceil(rows / 4096));
  device.destroy();
});

it('parameter helper packs the ridge lambda', () => {
  expect(Array.from(getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 2.5}))).toEqual([2.5]);
  expect(() =>
    getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda: 1}, new Float32Array(0))
  ).toThrow(/hold/);
});

it('chi-square survival matches known critical values and the df = 2 closed form', () => {
  expect(getChiSquareSurvival(3.841458820694124, 1)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(5.991464547107979, 2)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(7.814727903251179, 3)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(11.070497693516351, 5)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(18.307038053275146, 10)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(6.634896601021213, 1)).toBeCloseTo(0.01, 10);
  expect(getChiSquareSurvival(0, 4)).toBe(1);
  expect(getChiSquareSurvival(NaN, 4)).toBeNaN();
  for (const value of [0.1, 1, 2.5, 7, 30]) {
    expect(getChiSquareSurvival(value, 2)).toBeCloseTo(Math.exp(-value / 2), 12);
  }
  expect(getLogGammaOfHalfInteger(1)).toBeCloseTo(0.5 * Math.log(Math.PI), 14);
  expect(getLogGammaOfHalfInteger(6)).toBeCloseTo(Math.log(2), 14);
  expect(() => getLogGammaOfHalfInteger(0)).toThrow(/degrees/);
});

it('oracle reproduces a textbook simple regression', () => {
  const result = fitOrdinaryLeastSquaresOnCPU({
    predictors: Float32Array.from([1, 2, 3, 4, 5]),
    response: Float32Array.from([2, 4, 5, 4, 5]),
    predictorCount: 1
  });
  expect(result.status).toBe(0);
  expect(result.coefficients[0]).toBeCloseTo(2.2, 12);
  expect(result.coefficients[1]).toBeCloseTo(0.6, 12);
  expect(result.summary[1]).toBeCloseTo(0.6, 12);
  expect(result.summary[11]).toBeCloseTo(2.4, 12);
  expect(result.summary[3]).toBeCloseTo(0.8, 12);
  expect(result.standardErrors[1]).toBeCloseTo(Math.sqrt(0.08), 12);
  expect(result.standardErrors[0]).toBeCloseTo(Math.sqrt(0.88), 12);
  expect(result.tStatistics[1]).toBeCloseTo(0.6 / Math.sqrt(0.08), 12);
  expect(result.summary[2]).toBeCloseTo(1 - (0.4 * 4) / 3, 12);
});

it('oracle recovers known coefficients, exact fits, and reports singular and short designs', () => {
  const exact = createOrdinaryLeastSquaresScene(1, 200, 2, [3, -1], 0);
  const exactFit = fitOrdinaryLeastSquaresOnCPU(exact);
  expect(exactFit.status).toBe(0);
  expect(exactFit.summary[1]).toBeGreaterThan(1 - 1e-9);
  expect(exactFit.coefficients[0]).toBeCloseTo(2, 4);
  expect(exactFit.coefficients[1]).toBeCloseTo(3, 6);
  expect(exactFit.coefficients[2]).toBeCloseTo(-1, 6);

  const noisy = createOrdinaryLeastSquaresScene(2, 4000, 2, [3, -1], 0.5);
  const noisyFit = fitOrdinaryLeastSquaresOnCPU(noisy);
  expect(Math.abs(noisyFit.coefficients[1] - 3)).toBeLessThan(0.05);
  expect(Math.abs(noisyFit.coefficients[2] + 1)).toBeLessThan(0.05);
  // Residuals are near normal and homoskedastic: large p-values on average.
  expect(noisyFit.summary[8]).toBeGreaterThan(0);
  expect(noisyFit.summary[1]).toBeGreaterThan(0.9);

  const duplicate = createOrdinaryLeastSquaresScene(3, 50, 1, [2, 0], 0.1);
  for (let row = 0; row < 50; row++) {
    duplicate.predictors[row * 2 + 1] = duplicate.predictors[row * 2];
  }
  expect(fitOrdinaryLeastSquaresOnCPU(duplicate).status).toBe(1);

  const short = createOrdinaryLeastSquaresScene(4, 3, 1, [2, 1], 0.1);
  const shortFit = fitOrdinaryLeastSquaresOnCPU(short);
  expect(shortFit.status).toBe(2);
  expect(shortFit.summary[0]).toBe(3);
});

it('oracle excludes masked and non-finite rows and applies ridge shrinkage', () => {
  const scene = createOrdinaryLeastSquaresScene(5, 300, 1, [2, -3], 0.2, {
    withMask: true,
    withNonFinite: true
  });
  const fit = fitOrdinaryLeastSquaresOnCPU(scene);
  // 60 masked rows (row % 5 == 3), plus rows 1, 7, 298 excluded unless already masked.
  const excluded = new Set<number>([1, 7, 298]);
  for (let row = 0; row < 300; row++) {
    if (row % 5 === 3) {
      excluded.add(row);
    }
  }
  expect(fit.rowCount).toBe(300 - excluded.size);
  expect(Number.isNaN(fit.residuals[1])).toBe(true);
  expect(Number.isNaN(fit.residuals[3])).toBe(true);
  expect(Number.isNaN(fit.residuals[0])).toBe(false);
  const ridge = fitOrdinaryLeastSquaresOnCPU({...scene, ridgeLambda: 5000});
  expect(Math.abs(ridge.coefficients[1])).toBeLessThan(Math.abs(fit.coefficients[1]));
  expect(ridge.summary[15]).toBe(5000);
  expect(ridge.summary[11]).toBeGreaterThan(fit.summary[11]);
});

it('DEF-11 infers only an exact nonempty predictor layout and preserves explicit counts', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {predictorCount: undefined});
    const operation = new GPUOrdinaryLeastSquares(props);
    expect(operation.props.predictorCount).toBe(2);
    expect(
      () =>
        new GPUOrdinaryLeastSquares({
          ...props,
          predictors: createTransientView(graph, 'ambiguous', 'float32', 39)
        })
    ).toThrow(/predictorCount/);
    expect(() => new GPUOrdinaryLeastSquares({...props, predictorCount: 1})).toThrow(
      /predictors length/
    );
  } finally {
    device.destroy();
  }
});

it('CMP-11 returns a single design column unchanged and validates ordered columns', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const first = createTransientView(graph, 'first-column', 'float32', 5);
    const second = createTransientView(graph, 'second-column', 'float32', 5);
    expect(GPUOrdinaryLeastSquares.createDesign(graph, [first])).toBe(first);
    const predictors = GPUOrdinaryLeastSquares.createDesign(graph, [first, second]);
    expect(predictors.length).toBe(10);
    expect(predictors.format).toBe('float32');
    expect(() => GPUOrdinaryLeastSquares.createDesign(graph, [])).toThrow(/columns/);
    expect(() =>
      GPUOrdinaryLeastSquares.createDesign(graph, [
        first,
        createTransientView(graph, 'short-column', 'float32', 4)
      ])
    ).toThrow(/length/);
  } finally {
    device.destroy();
  }
});

import {describeOutputs} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operator-output-declarations';

it('UCO-17 audited OLS allocation plan matches active and maximum constructor lengths', () => {
  const plan = describeOutputs(
    {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUOrdinaryLeastSquares'
    },
    'default',
    {rowCount: 10, predictorCount: 1}
  );
  expect(plan.declaration).toBe('declared');
  for (const capacityKind of ['minimumCapacity', 'capacity'] as const) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const getView = <Format extends 'float32' | 'uint32'>(path: string, format: Format) => {
        const declaration = plan.outputs.find(output => output.path === `output.${path}`);
        if (!declaration) throw new Error(`Missing output ${path}`);
        expect(declaration.format).toBe(format);
        return createTransientView(graph, path, format, declaration[capacityKind]);
      };
      const output = {
        coefficients: getView('coefficients', 'float32'),
        standardErrors: getView('standardErrors', 'float32'),
        tStatistics: getView('tStatistics', 'float32'),
        summary: getView('summary', 'float32'),
        status: {code: getView('status.code', 'uint32')},
        fitted: getView('fitted', 'float32'),
        residuals: getView('residuals', 'float32'),
        predictions: getView('predictions', 'float32')
      };
      expect(
        () =>
          new GPUOrdinaryLeastSquares({
            predictors: createTransientView(graph, 'x', 'float32', 10),
            response: createTransientView(graph, 'y', 'float32', 10),
            predictorCount: 1,
            output
          })
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  }
  for (const name of ['coefficients', 'standardErrors', 'tStatistics']) {
    expect(plan.outputs.find(output => output.path === `output.${name}`)).toMatchObject({
      capacity: 16,
      minimumCapacity: 2,
      activeLength: 2,
      format: 'float32',
      status: ['output.status.code']
    });
  }
  expect(plan.outputs.find(output => output.path === 'output.summary')).toMatchObject({
    capacity: GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
    format: 'float32'
  });
  expect(plan.outputs.find(output => output.path === 'output.status.code')).toMatchObject({
    capacity: 1,
    format: 'uint32'
  });
});

it('W3-04 uses the shared tile default and preserves an explicit width', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (const [tileRowCount, expectedWidth, expectedCount] of [
      [undefined, 64, 1],
      [3, 3, 7],
      [100, 100, 1]
    ] as const) {
      const fit = new GPUOrdinaryLeastSquares(createProps(graph, {tileRowCount}));
      expect({tileRowCount: fit.tileRowCount, tileCount: fit.tileCount}).toEqual({
        tileRowCount: expectedWidth,
        tileCount: expectedCount
      });
    }
  } finally {
    device.destroy();
  }
});

it('CMP-01 adds the completeness gate before OLS and preserves loss view identities', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {id: 'joined-fit'});
    const matched = createTransientView(graph, 'matched', 'uint32', 20);
    const complete = createTransientView(graph, 'complete', 'uint32', 1);
    const candidateOverflow = createTransientView(graph, 'candidate-overflow', 'uint32', 1);
    const joinedTable = {matched, status: {complete, candidateOverflow}};
    const composition = GPUOrdinaryLeastSquares.addJoinedTableFit(graph, props, joinedTable);
    expect(composition.fit.props.mask).toBe(composition.mask);
    expect(composition.mask.length).toBe(20);
    expect(composition.status.complete).toBe(complete);
    expect(composition.status.candidateOverflow).toBe(candidateOverflow);
    expect(composition.status).not.toBe(joinedTable.status);
  } finally {
    device.destroy();
  }
});

it('B1 composition kernels use the public graph add hook', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const add = vi.spyOn(graph, 'add');
    const columns = [
      createTransientView(graph, 'hook-x', 'float32', 20),
      createTransientView(graph, 'hook-z', 'float32', 20)
    ];
    GPUOrdinaryLeastSquares.createDesign(graph, columns, 'hook-design');
    expect(add.mock.calls.map(([node]) => ('id' in node ? node.id : undefined))).toContain(
      'hook-design-row-indices'
    );
    const props = createProps(graph, {id: 'hook-fit'});
    GPUOrdinaryLeastSquares.addJoinedTableFit(graph, props, {
      matched: createTransientView(graph, 'hook-matched', 'uint32', 20),
      status: {complete: createTransientView(graph, 'hook-complete', 'uint32', 1)}
    });
    expect(add.mock.calls.map(([node]) => ('id' in node ? node.id : undefined))).toContain(
      'hook-fit-join-training-mask'
    );
  } finally {
    device.destroy();
  }
});

for (const invalid of ['matched', 'output'] as const) {
  it(`B3 rejected ${invalid} declaration can be corrected on the same graph and ID`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, {id: 'retry-fit'});
      const joined = {
        matched: createTransientView(graph, 'retry-matched', 'uint32', 20),
        status: {complete: createTransientView(graph, 'retry-complete', 'uint32', 1)}
      };
      const badProps =
        invalid === 'output'
          ? {
              ...props,
              output: {
                ...props.output,
                coefficients: createTransientView(graph, 'retry-short', 'float32', 1)
              }
            }
          : props;
      const badJoined =
        invalid === 'matched'
          ? {...joined, matched: createTransientView(graph, 'retry-long', 'uint32', 21)}
          : joined;
      expect(() => GPUOrdinaryLeastSquares.addJoinedTableFit(graph, badProps, badJoined)).toThrow();
      expect(() => GPUOrdinaryLeastSquares.addJoinedTableFit(graph, props, joined)).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

it('B4 joined fit captures inherited getters once', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const reads = {predictors: 0, response: 0, output: 0};
    class InheritedProps {
      get predictors() {
        reads.predictors++;
        return props.predictors;
      }
      get response() {
        reads.response++;
        return props.response;
      }
      get output() {
        reads.output++;
        return props.output;
      }
    }
    const result = GPUOrdinaryLeastSquares.addJoinedTableFit(graph, new InheritedProps(), {
      matched: createTransientView(graph, 'inherited-matched', 'uint32', 20),
      status: {complete: createTransientView(graph, 'inherited-complete', 'uint32', 1)}
    });
    expect(result.fit.props.predictors).toBe(props.predictors);
    expect(reads).toEqual({predictors: 1, response: 1, output: 1});
  } finally {
    device.destroy();
  }
});

it('B5 normalizes the response before any centered subtraction and restores caller units last', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {id: 'scaled-response'});
    const fit = new GPUOrdinaryLeastSquares(props);
    const nodes = fit.getCommandNodes(graph);
    const identifiers = nodes.map(node => node.id);
    expect(identifiers).toContain('scaled-response-response-normalize');
    expect(identifiers.indexOf('scaled-response-response-normalize')).toBeLessThan(
      identifiers.indexOf('scaled-response-means-origins')
    );
    expect(identifiers.at(-1)).toBe('scaled-response-response-restore-report');
    expect(nodes.length).toBe(fit.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});
