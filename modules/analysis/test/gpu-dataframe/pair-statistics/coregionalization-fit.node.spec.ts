// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeCoregionalizationFitOnCPU,
  fitLinearModelOfCoregionalization
} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import {GPUCoregionalizationFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {getGPULocalKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {createCoregionalizationAuditProps} from '../../analysis-core/kernel-audit/cases-coregionalization-fit';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import fixtures from './fixtures/coregionalization-numpy.json';
import maternShapeOracle from './fixtures/matern-shapes-decimal.json';
import {
  getVariogramModelShape,
  type VariogramModelType
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';

for (const fixture of fixtures) {
  it(`matches the independent NumPy Goulard–Voltz oracle: ${fixture.name}`, () => {
    const result = computeCoregionalizationFitOnCPU(
      {
        distances: Float32Array.from(fixture.distances),
        semivariances: Float32Array.from(fixture.semivariances),
        pairCounts: Uint32Array.from(fixture.pairCounts),
        parameters: Float32Array.from(fixture.parameters)
      },
      fixture
    );
    expect(result.validity[0]).toBe(fixture.validity);
    expect(result.converged[0]).toBe(fixture.converged);
    expect(result.iterationCount[0]).toBe(fixture.iterationCount);
    expect(result.invalidCount[0]).toBe(fixture.invalidCount);
    expect(result.iterationLimitReached[0]).toBe(
      Number(Boolean(fixture.validity && !fixture.converged))
    );
    if (!fixture.matrices) {
      expect(Array.from(result.matrices).every(Number.isNaN)).toBe(true);
      expect(Array.from(result.loadings).every(Number.isNaN)).toBe(true);
      return;
    }
    fixture.matrices.forEach((expected, index) =>
      expect(result.matrices[index]).toBeCloseTo(expected, 5)
    );
    expect(result.diagnostics[0]).toBeCloseTo(fixture.residual!, 6);
    expect(result.diagnostics[1]).toBeCloseTo(fixture.relativeChange!, 6);
    expect(result.diagnostics[2]).toBe(fixture.distances.length - fixture.invalidCount);
    expect(result.diagnostics[3]).toBe(
      fixture.pairCounts.reduce(
        (total, count, index) => total + (fixture.distances[index] > 0 ? count : 0),
        0
      )
    );
    const variableCount = fixture.variableCount;
    for (let structure = 0; structure < fixture.structureCount; structure++) {
      for (let row = 0; row < variableCount; row++) {
        for (let column = 0; column < variableCount; column++) {
          let reconstructed = 0;
          for (let component = 0; component < variableCount; component++)
            reconstructed +=
              result.loadings[structure * variableCount ** 2 + row * variableCount + component] *
              result.loadings[structure * variableCount ** 2 + column * variableCount + component];
          expect(reconstructed).toBeCloseTo(
            result.matrices[structure * variableCount ** 2 + row * variableCount + column],
            5
          );
        }
      }
    }
  });
}

const input = {
  distances: new Float32Array([1]),
  semivariances: new Float32Array([2, -1, 3]),
  pairCounts: new Uint32Array([10]),
  parameters: getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 1, sill: 1}])
};

it('returns loadings accepted by the existing supplied-LMC parameter packer', () => {
  const result = fitLinearModelOfCoregionalization(input, {
    variableCount: 2,
    structures: [{model: 'spherical', range: 1, sill: 1}]
  });
  expect(result.validity[0]).toBe(1);
  expect(result.coregionalizationModels).toHaveLength(1);
  const parameters = getGPULocalKrigingParameterValues({
    variableCount: 2,
    coregionalizationModels: result.coregionalizationModels
  });
  expect(Array.from(parameters.subarray(8))).toEqual(Array.from(result.matrices));
});

it('builds the documented fitted-LMC local kriging graph with its required covariance prop', () => {
  const fit = fitLinearModelOfCoregionalization(input, {
    variableCount: 2,
    structures: [{model: 'spherical', range: 1, sill: 1}]
  });
  expect(fit.validity[0]).toBe(1);
  const context = createKernelAuditContext('fitted-lmc-local-kriging');
  try {
    const contributor = new GPULocalKriging({
      positions: context.view('float32x2', 2),
      observations: context.view('float32x4', 2),
      supports: context.view('float32x2', 1),
      mode: 'co-kriging',
      coKrigingMean: 'known',
      variableCount: 2,
      neighborCount: 2,
      covariance: fit.coregionalizationModels[0].covariance,
      coregionalizationModels: fit.coregionalizationModels,
      output: {
        values: context.view('float32x4', 1),
        status: context.view('uint32', 1)
      }
    });
    expect(
      getGPUCommandNodeKernelSources(contributor.getCommandNodes(context.graph)).length
    ).toBeGreaterThan(0);
  } finally {
    context.device.destroy();
  }
});

for (const range of [1e8, 1e9]) {
  it(`returns a usable approximately identity supplied Gaussian LMC at lag/range = 1/${range}`, () => {
    const semivariance = Math.fround(3 / range ** 2);
    const fit = fitLinearModelOfCoregionalization(
      {
        distances: new Float32Array([1]),
        semivariances: new Float32Array([semivariance, 0, semivariance]),
        pairCounts: new Uint32Array([1])
      },
      {variableCount: 2, structures: [{model: 'gaussian', range, sill: 1}]}
    );
    expect(fit.validity[0]).toBe(1);
    expect(fit.converged[0]).toBe(1);
    expect(fit.iterationLimitReached[0]).toBe(0);
    expect(fit.coregionalizationModels).toHaveLength(1);
    // gamma(h) = 3(h/range)^2 + O((h/range)^4); the omitted relative term is < 2e-16.
    // Rounding the measured semivariance to float32 accounts for the deviation from identity.
    const expectedCoefficient = semivariance / (3 / range ** 2);
    [expectedCoefficient, 0, 0, expectedCoefficient].forEach((expected, index) => {
      expect(Math.abs(fit.matrices[index] - expected)).toBeLessThan(1e-6);
    });
    expect(fit.coregionalizationModels[0].loadings.every(Number.isFinite)).toBe(true);
  });
}

it('skips whole NaN/infinite lag matrices and zero-count padding', () => {
  const result = computeCoregionalizationFitOnCPU(
    {
      ...input,
      distances: new Float32Array([1, 2, 3, Infinity]),
      pairCounts: new Uint32Array([10, 4, 0, 5]),
      semivariances: new Float32Array([2, -1, 3, 9, NaN, 8, 999, 999, 999, 1, 1, 1])
    },
    {variableCount: 2, structureCount: 1}
  );
  expect(result.invalidCount[0]).toBe(3);
  expect(Array.from(result.matrices)).toEqual([2, -1, -1, 3]);
});

it('reports invalid parameters and zero-energy structures without a convergence claim', () => {
  for (const parameters of [
    new Float32Array([0, 0, 0, 1, 1]),
    new Float32Array([1e-5, 5, 0, 1, 1]),
    new Float32Array([1e-5, 0, 0, 1, Infinity]),
    new Float32Array([1e-5, 0, 0, 0, 1])
  ]) {
    const result = computeCoregionalizationFitOnCPU(
      {...input, parameters},
      {variableCount: 2, structureCount: 1}
    );
    expect(result.validity[0]).toBe(0);
    expect(result.converged[0]).toBe(0);
    expect(result.iterationLimitReached[0]).toBe(0);
  }
});

it('rejects malformed arrays and unsupported dimensions by name', () => {
  expect(() =>
    computeCoregionalizationFitOnCPU(
      {...input, semivariances: new Float32Array(2)},
      {variableCount: 2, structureCount: 1}
    )
  ).toThrow(/semivariances.length/);
  expect(() =>
    computeCoregionalizationFitOnCPU(
      {...input, pairCounts: new Uint32Array(0)},
      {variableCount: 2, structureCount: 1}
    )
  ).toThrow(/pairCounts.length/);
  expect(() =>
    computeCoregionalizationFitOnCPU(
      {...input, parameters: new Float32Array(2)},
      {variableCount: 2, structureCount: 1}
    )
  ).toThrow(/parameters.length/);
  expect(() =>
    getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 0, sill: 1}])
  ).toThrow(/coregionalization-fit structures/);
  expect(() =>
    getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 1, sill: 1}], 0)
  ).toThrow(/tolerance/);
  expect(() =>
    computeCoregionalizationFitOnCPU(input, {variableCount: 5, structureCount: 1})
  ).toThrow(/variableCount.*4/);
});

it('enforces GPU caps, graph topology, packed formats and node budget before emitting nodes', () => {
  const context = createKernelAuditContext('coregionalization-limits');
  try {
    const props = createCoregionalizationAuditProps(context);
    for (const [property, value] of [
      ['variableCount', 5],
      ['structureCount', 5],
      ['maximumIterations', 1025],
      ['maximumIterations', -1]
    ] as const) {
      expect(() => new GPUCoregionalizationFit({...props, [property]: value})).toThrow(
        new RegExp(`${property}.*(4|1024|0)`)
      );
    }
    expect(
      () => new GPUCoregionalizationFit({...props, distances: context.view('float32', 257)})
    ).toThrow(/distances.length.*256/);
    expect(
      () => new GPUCoregionalizationFit({...props, semivariances: context.view('float32', 3)})
    ).toThrow(/semivariances.length/);
    expect(
      () =>
        new GPUCoregionalizationFit({
          ...props,
          output: {...props.output, loadings: props.output.matrices}
        })
    ).toThrow(/distinct buffers/);
    const contributor = new GPUCoregionalizationFit({...props, maximumIterations: 2});
    const nodes = contributor.getCommandNodes(context.graph);
    expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const sweepNodes = nodes.filter(node => node.id.includes('-sweep-'));
    expect(sweepNodes).toHaveLength(2);
    expect(
      sweepNodes.every(
        node => node.condition?.source === 'gpu' && node.condition.mode === 'indirect'
      )
    ).toBe(true);
    expect(
      kernels.some(kernel => kernel.source.includes('dispatch[dispatchOffset + 0u] = 0u'))
    ).toBe(true);
  } finally {
    context.device.destroy();
  }
});

it('matches independent 80-digit Decimal Matérn shapes at tiny lags, the series transition and long lags', () => {
  for (const reference of maternShapeOracle) {
    const shape = getVariogramModelShape(
      reference.model as VariogramModelType,
      reference.scaledDistance
    );
    if (reference.shape === 0) expect(shape).toBe(0);
    else
      expect(
        Math.abs(shape / reference.shape - 1),
        `${reference.model} at ${reference.scaledDistance}`
      ).toBeLessThan(2e-14);
  }
});

for (const fixture of fixtures.filter(
  entry => entry.name.startsWith('matern-') && entry.name.includes('tiny-lag')
)) {
  it(`returns a valid approximately identity supplied LMC for ${fixture.name}`, () => {
    const fit = fitLinearModelOfCoregionalization(
      {
        distances: Float32Array.from(fixture.distances),
        semivariances: Float32Array.from(fixture.semivariances),
        pairCounts: Uint32Array.from(fixture.pairCounts)
      },
      {
        variableCount: 2,
        structures: [
          {
            model: fixture.parameters[1] === 3 ? 'matern-3/2' : 'matern-5/2',
            range: fixture.parameters[4],
            sill: 1
          }
        ]
      }
    );
    expect(fit.validity[0]).toBe(1);
    expect(fit.converged[0]).toBe(1);
    expect(fit.iterationLimitReached[0]).toBe(0);
    expect(fit.coregionalizationModels).toHaveLength(1);
    fixture.matrices!.forEach((expected, index) =>
      expect(Math.abs(fit.matrices[index] - expected)).toBeLessThan(1e-6)
    );
    [1, 0, 0, 1].forEach((expected, index) =>
      expect(Math.abs(fit.matrices[index] - expected)).toBeLessThan(1e-6)
    );
  });
}
