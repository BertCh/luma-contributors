// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import type {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeSpatialErrorGMOnCPU,
  CPUSpatialErrorGMOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import {
  GPUSpatialErrorGMOperation,
  type SpatialErrorGMOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation';
import {getGPUSpatialWeightsTracesParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {SPREG_ERROR_GM_REFERENCE, SPREG_ERROR_GM_HOM_REFERENCE} from './spatial-error-gm-reference';
import {createErrorScene} from './spatial-error-gm-oracle';
import fixtures from './spatial-error-gm-operation-fixtures.json';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

function getInput(name: 'lattice' | 'knn' | 'binary' = 'knn') {
  if (name === 'binary') return createErrorScene(324, 9, 0.5, 'lattice-binary');
  const fixture = SPREG_ERROR_GM_REFERENCE[name];
  return {
    weights: {
      offsets: Uint32Array.from(fixture.offsets),
      neighbors: Uint32Array.from(fixture.neighbors),
      weights: Float32Array.from(fixture.weights)
    },
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response)
  };
}

it('Kelejian-Prucha ignores a zero trace probe count at construction, lowering and computation', () => {
  const input = {
    weights: {
      offsets: Uint32Array.of(0, 3, 6, 9, 12),
      neighbors: Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2),
      weights: new Float32Array(12).fill(1)
    },
    predictors: Float32Array.of(-1, 0, 1, 2),
    response: Float32Array.of(0, 1, 4, 9)
  };
  const props: SpatialErrorGMOperationProps = {
    id: 'ignored-gm-traces',
    rowCount: 4,
    predictorCount: 1,
    estimator: 'kelejian-prucha',
    traceProbeCount: 0,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    predictors: 'predictors',
    response: 'response',
    output: {status: {code: 'code'}, table: 'table', summary: 'summary', residuals: 'residuals'}
  };
  expect(() => new CPUSpatialErrorGMOperation(props)).not.toThrow();
  const result = computeSpatialErrorGMOnCPU(input, props);
  // W=J-I, u=[1,-1,-1,1], Wu=-u. The projected moment criterion is
  // 1.4*(1+lambda)^4 on [-0.99/3,0.99/3], so its minimum is the lower endpoint.
  expect(result.status[0]).toBe(0);
  expectRegressionRowsClose(
    [result.table[0], result.table[4], result.table[8]],
    [2, 3, -0.33],
    1e-7
  );
  expectRegressionRowsClose(result.residuals, [1, -1, -1, 1], 1e-7);
  expectRegressionRowsClose(
    [result.summary[2], result.summary[5], result.summary[10]],
    [0.67 ** 2, 1.4 * 0.67 ** 4, 0.33],
    1e-7
  );
  expect(result.summary[11]).toBe(0);
  expect(result.summary[12]).toBe(1);
  const ignoredMode = computeSpatialErrorGMOnCPU(input, {...props, traceMode: 'unused' as never});
  expectRegressionRowsClose(ignoredMode.table, result.table, 0);
  expect(() => new CPUSpatialErrorGMOperation({...props, estimator: 'homoskedastic'})).toThrow(
    /traceProbeCount/
  );
  // Unlike the ignored trace settings, a malformed packed seed view must still throw.
  expect(() =>
    computeSpatialErrorGMOnCPU({...input, traceParameters: new Uint32Array(0)}, props)
  ).toThrow(/traceParameters/);
  const device = createNullWebGPUDevice();
  const operation = new GPUSpatialErrorGMOperation(props);
  try {
    const graph = new GPUCommandGraph(device);
    const shapes = {
      offsets: {format: 'uint32' as const, length: 5},
      neighbors: {format: 'uint32' as const, length: 12},
      weights: {format: 'float32' as const, length: 12},
      predictors: {format: 'float32' as const, length: 4},
      response: {format: 'float32' as const, length: 4}
    };
    const views = Object.fromEntries(
      Object.entries({...shapes, ...operation.outputs}).map(([name, shape]) => [
        name,
        createTransientView(graph, name, shape.format, shape.length)
      ])
    );
    const contributor = operation.createContributor(
      name => views[name] as never
    ) as GPUSpatialErrorGM;
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    operation.destroy();
    device.destroy();
  }
});

for (const name of ['lattice', 'knn', 'binary'] as const) {
  for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const) {
    it(`matches committed spreg 1.9.1 ${estimator} output and NumPy conditioning on ${name}`, () => {
      const input = getInput(name);
      const result = computeSpatialErrorGMOnCPU(input, {predictorCount: 2, estimator});
      const fixture =
        SPREG_ERROR_GM_HOM_REFERENCE[name][
          estimator === 'homoskedastic' ? 'hom' : 'kelejianPrucha'
        ];
      expect(result.status[0]).toBe(0);
      // spreg's local optimizer stops earlier than the global quartic minimization.
      expect(Math.abs(result.summary[1] - fixture.lambda)).toBeLessThan(5e-4);
      fixture.betas.forEach((value, row) => {
        expect(Math.abs(result.table[row * 4] - value) / fixture.standardErrors[row]).toBeLessThan(
          5e-3
        );
        expectRegressionRowsClose(
          result.table.subarray(row * 4 + 1, row * 4 + 4),
          Float32Array.of(fixture.standardErrors[row], fixture.z[row], fixture.p[row]),
          0.015
        );
      });
      expectRegressionRowsClose(
        [result.summary[2], result.summary[3]],
        [fixture.sigma2, fixture.pseudoRSquared],
        name === 'binary' ? 3e-3 : 1e-3
      );
      expectRegressionRowsClose(
        result.summary.subarray(6, 9),
        Object.values(fixture.conditioning),
        name === 'binary' ? 3e-3 : 1e-3
      );
      expect(result.summary[0]).toBe(input.response.length);
      expect(result.summary[9]).toBe(0);
      expect(result.summary[11]).toBe(0);
      expect(result.summary[12]).toBe(0);
      if (estimator === 'homoskedastic') {
        expectRegressionRowsClose(
          result.table.subarray(13),
          [fixture.standardErrors[3], fixture.z[3], fixture.p[3]],
          name === 'binary' ? 5e-3 : 1e-3
        );
        expect(result.table[15]).toBeGreaterThan(0);
      } else expect(result.table.subarray(13).every(Number.isNaN)).toBe(true);
      // Residual reconstruction is an independent identity using the reported coefficients.
      const reconstructed = Array.from(
        input.response,
        (value, row) =>
          value -
          result.table[0] -
          input.predictors[row * 2] * result.table[4] -
          input.predictors[row * 2 + 1] * result.table[8]
      );
      for (let row = 0; row < reconstructed.length; row++)
        expect(Math.abs(result.residuals[row] - reconstructed[row])).toBeLessThan(3e-6);
      expectRegressionRowsClose(
        [result.summary[4]],
        [result.residuals.reduce((sum, value) => sum + value * value, 0)],
        2e-7
      );
    });
  }
}

for (const fixture of fixtures) {
  it(`matches dense NumPy/SciPy matrices, SVD and bounded search for every output: ${fixture.name}`, () => {
    const result = computeSpatialErrorGMOnCPU(
      {
        weights: {
          offsets: Uint32Array.from(fixture.weights.offsets),
          neighbors: Uint32Array.from(fixture.weights.neighbors),
          weights: Float32Array.from(fixture.weights.weights)
        },
        predictors: Float32Array.from(fixture.predictors),
        response: Float32Array.from(fixture.response)
      },
      {
        predictorCount: fixture.predictorCount,
        estimator: fixture.estimator as 'homoskedastic' | 'kelejian-prucha'
      }
    );
    expect(result.status[0]).toBe(0);
    for (const port of ['table', 'summary', 'residuals'] as const)
      expectRegressionRowsClose(
        result[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        1e-5
      );
  });
}

it('retains offsets, raw weight scaling, exact/stochastic mode and the registered seed', () => {
  const input = getInput();
  for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const) {
    const options = {predictorCount: 2, estimator};
    const ordinary = computeSpatialErrorGMOnCPU(input, options);
    // Response units leave either estimator's lambda unchanged. Scaling W changes the
    // relative weighting of the moments, so raw weights are checked against spreg above.
    const scaled = computeSpatialErrorGMOnCPU(
      {...input, response: Float32Array.from(input.response, value => value * 2)},
      options
    );
    expectRegressionRowsClose(
      [scaled.summary[1], scaled.summary[2] / 4],
      [ordinary.summary[1], ordinary.summary[2]],
      1e-6
    );
    expectRegressionRowsClose(
      Float32Array.from(scaled.residuals, value => value / 2),
      ordinary.residuals,
      1e-5
    );
  }
  const stochastic = (seed: number) =>
    computeSpatialErrorGMOnCPU(
      {...input, traceParameters: getGPUSpatialWeightsTracesParameterValues({seed})},
      {predictorCount: 2, traceMode: 'stochastic', traceProbeCount: 16}
    );
  const first = stochastic(42);
  expect(first.status[0]).toBe(0);
  expect(first.summary[11]).toBe(1);
  expect(Array.from(first.table)).toEqual(Array.from(stochastic(42).table));
  expect(first.summary[1]).not.toBe(stochastic(43).summary[1]);
});

it('reports singular, too-small and empty-weights cases, and rejects inconsistent arrays', () => {
  const input = getInput();
  expect(
    computeSpatialErrorGMOnCPU(
      {...input, predictors: new Float32Array(input.predictors.length)},
      {predictorCount: 2}
    ).status[0]
  ).toBe(1);
  const empty = computeSpatialErrorGMOnCPU(
    {...input, weights: {...input.weights, offsets: new Uint32Array(input.weights.offsets.length)}},
    {predictorCount: 2}
  );
  expect(empty.status[0]).toBe(3);
  expect(empty.table.every(Number.isNaN)).toBe(true);
  expect(empty.summary[0]).toBe(input.response.length);
  expect(empty.summary.subarray(1).every(Number.isNaN)).toBe(true);
  expect(empty.residuals.every(Number.isNaN)).toBe(true);
  for (const changes of [
    {predictors: new Float32Array(1)},
    {response: new Float32Array(1)},
    {traceParameters: new Uint32Array(0)}
  ])
    expect(() => computeSpatialErrorGMOnCPU({...input, ...changes}, {predictorCount: 2})).toThrow(
      new RegExp(Object.keys(changes)[0])
    );
  expect(() =>
    computeSpatialErrorGMOnCPU(
      {...input, weights: {...input.weights, offsets: Uint32Array.of(0, 999999)}},
      {predictorCount: 2}
    )
  ).toThrow(/offsets/);
  expect(() =>
    computeSpatialErrorGMOnCPU(
      {...input, weights: {...input.weights, weights: new Float32Array(1)}},
      {predictorCount: 2}
    )
  ).toThrow(/weights/);
  const tooSmall = computeSpatialErrorGMOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1)
      },
      predictors: Float32Array.of(0, 1),
      response: Float32Array.of(2, 3)
    },
    {predictorCount: 1}
  );
  expect(tooSmall.status[0]).toBe(2);
  const failedReport = computeSpatialErrorGMOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4),
        neighbors: Uint32Array.of(1, 2, 3, 0),
        weights: Float32Array.of(1, 1, 1, 1)
      },
      predictors: Float32Array.of(0, 0, 1, 0, 0, 1, 1, 1),
      response: Float32Array.of(0, 0, 0, 1)
    },
    {predictorCount: 2}
  );
  expect(failedReport.status[0]).toBe(2);
  expect(failedReport.table.every(Number.isNaN)).toBe(true);
  // The GPU finish failure clears the report, but does not clear the earlier residual pass.
  expect(failedReport.residuals.every(Number.isFinite)).toBe(true);
});

it('validates both operation constructions and preserves the graph contract', () => {
  const input = getInput();
  const props: SpatialErrorGMOperationProps = {
    id: 'error-gm-test',
    rowCount: input.response.length,
    predictorCount: 2,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    predictors: 'predictors',
    response: 'response',
    output: {status: {code: 'code'}, table: 'table', summary: 'summary', residuals: 'residuals'}
  };
  const shapes = {
    offsets: {format: 'uint32' as const, length: input.weights.offsets.length},
    neighbors: {format: 'uint32' as const, length: input.weights.neighbors.length},
    weights: {format: 'float32' as const, length: input.weights.weights.length},
    predictors: {format: 'float32' as const, length: input.predictors.length},
    response: {format: 'float32' as const, length: input.response.length}
  };
  for (const changes of [{predictorCount: 0}, {tileRowCount: 0}, {estimator: 'invalid' as never}]) {
    expect(() => new CPUSpatialErrorGMOperation({...props, ...changes})).toThrow(/error-gm-test/);
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialErrorGMOperation({...props, ...changes}),
        shapes
      )
    ).toThrow(/error-gm-test/);
  }
  for (const Operation of [GPUSpatialErrorGMOperation, CPUSpatialErrorGMOperation])
    expect(() => new Operation({...props, output: {...props.output, table: 'response'}})).toThrow(
      /error-gm-test/
    );
  expect(new CPUSpatialErrorGMOperation(props).contract.outputs).toEqual(
    new GPUSpatialErrorGMOperation(props).contract.outputs
  );
  expect(() =>
    constructRegressionOperationContributor(new GPUSpatialErrorGMOperation(props), shapes)
  ).not.toThrow();
});
