// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSeeminglyUnrelatedRegressionOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import seeded from './sur-review-fixtures.json';
import {expectIteratedSURResult, type IteratedSURExpectation} from './iterated-sur-test-utils';
import {CPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation';
import {GPUSeeminglyUnrelatedRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation';
import spatialFixture from './seemingly-unrelated-regression-spatial-operation-fixtures.json';
import {createTransientView, GPUCommandGraph, GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getSpatialSURWeightValidationNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-sur-nodes';
import {GPUSeeminglyUnrelatedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {getGPUSeeminglyUnrelatedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/seemingly-unrelated-regression-parameters';
import {getCPUVectorRows, makeCPUVector} from '../../utils/mixed-execution-test-utils';
import type {CPUVector} from '@luma.gl/graph/gpu-data';

it('R9 validates the public parameter layout and refuses physical parameter/output aliasing', () => {
  const target = Float32Array.of(99, 77);
  expect(getGPUSeeminglyUnrelatedRegressionParameterValues({tolerance: 0.5}, target)).toBe(target);
  expect(Array.from(target)).toEqual([0.5, 77]);
  expect(Array.from(getGPUSeeminglyUnrelatedRegressionParameterValues())).toEqual([
    0.000009999999747378752
  ]);
  expect(() => getGPUSeeminglyUnrelatedRegressionParameterValues({}, new Float32Array())).toThrow(
    /target.length/
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `parameter-layout-${serial++}`, format, length);
    const parameters = view('float32', 1);
    const status = graph.createDataView(parameters.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    expect(
      () =>
        new GPUSeeminglyUnrelatedRegression({
          equationCount: 2,
          predictorCount: 1,
          rowCount: 8,
          predictors: view('float32', 16),
          responses: view('float32', 16),
          parameters,
          iteration: {maximumIterations: 2},
          output: {
            coefficients: view('float32', 4),
            covariance: view('float32', 16),
            residualCovariance: view('float32', 4),
            residuals: view('float32', 16),
            status
          }
        })
    ).toThrow(/parameters.*status|status.*parameters/);
    const spatialParameters = view('float32', 2);
    const tolerance = graph.createDataView(spatialParameters.buffer, {
      format: 'float32',
      byteOffset: 0,
      length: 1
    });
    expect(
      () =>
        new GPUSeeminglyUnrelatedRegression({
          equationCount: 2,
          predictorCount: 1,
          rowCount: 8,
          predictors: view('float32', 16),
          responses: view('float32', 16),
          parameters: tolerance,
          iteration: {maximumIterations: 2},
          spatial: {
            weights: {
              offsets: view('uint32', 9),
              neighbors: view('uint32', 8),
              weights: view('float32', 8)
            },
            output: {parameters: spatialParameters, status: view('uint32', 2)}
          },
          output: {
            coefficients: view('float32', 4),
            covariance: view('float32', 16),
            residualCovariance: view('float32', 4),
            residuals: view('float32', 16),
            status: view('uint32', 1)
          }
        })
    ).toThrow(/overlap|alias|disjoint/);
  } finally {
    device.destroy();
  }
});
it('rejects the retired SUR iteration-count output on all construction forms', () => {
  const expected = /sur-test.*output\.iterationCount.*output\.iteration\.status\.iterationCount/;
  const operationProps = {
    id: 'sur-test',
    equationCount: 2,
    predictorCount: 1,
    rowCount: 8,
    predictors: 'predictors',
    responses: 'responses',
    output: {
      coefficients: 'coefficients',
      covariance: 'covariance',
      residualCovariance: 'residualCovariance',
      residuals: 'residuals',
      status: 'status',
      iterationCount: 'iterations'
    }
  };
  expect(() => new GPUSeeminglyUnrelatedRegressionOperation(operationProps)).toThrow(expected);
  expect(() => new CPUSeeminglyUnrelatedRegressionOperation(operationProps)).toThrow(expected);

  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let index = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `retired-sur-output-${index++}`, format, length);
    const producerProps = {
      ...operationProps,
      predictors: view('float32', 16),
      responses: view('float32', 16),
      output: {
        coefficients: view('float32', 4),
        covariance: view('float32', 16),
        residualCovariance: view('float32', 4),
        residuals: view('float32', 16),
        status: view('uint32', 1),
        iterationCount: view('uint32', 1)
      }
    };
    expect(() => new GPUSeeminglyUnrelatedRegression(producerProps)).toThrow(expected);
  } finally {
    device.destroy();
  }
});

it('R8 rejects stale preceding-round values in each spatial numeric port', () => {
  const fixture = seeded.spatialFinal;
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const) {
    expect(
      () =>
        expectIteratedSURResult(
          {...fixture.expected, [name]: fixture.previous[name]},
          fixture.expected,
          5e-4
        ),
      name
    ).toThrow();
  }
});

it('R8 CPU final spatial round matches the stronger independent three-equation oracle', () => {
  const fixture = seeded.spatialFinal;
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(fixture.predictors),
      responses: Float32Array.from(fixture.responses),
      spatial: {
        weights: {
          offsets: Uint32Array.from(fixture.weights.offsets),
          neighbors: Uint32Array.from(fixture.weights.neighbors),
          weights: Float32Array.from(fixture.weights.weights)
        }
      }
    },
    {
      equationCount: 3,
      predictorCount: 2,
      rowCount: 40,
      iteration: {maximumIterations: 100, tolerance: fixture.tolerance}
    }
  );
  expectIteratedSURResult(result, fixture.expected, 2e-6);
});

it('R9 replays one compiled CPU operation graph with two tolerances and invalid runtime values', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUGraph();
    for (const [name, length] of [
      ['predictors', 16],
      ['responses', 16],
      ['initial', 16],
      ['parameters', 1]
    ] as const)
      graph.addInput(name, gpuGraph.vector('float32', {length}));
    const output = {
      coefficients: 'coefficients',
      covariance: 'covariance',
      residualCovariance: 'residualCovariance',
      residuals: 'residuals',
      status: 'status',
      iteration: {status: {iterationCount: 'iterationCount'}}
    };
    graph.add(
      new CPUSeeminglyUnrelatedRegressionOperation({
        equationCount: 2,
        predictorCount: 1,
        rowCount: 8,
        predictors: 'predictors',
        responses: 'responses',
        initialResiduals: 'initial',
        parameters: 'parameters',
        iteration: {maximumIterations: 100, tolerance: 1e40},
        output
      })
    );
    const {iteration: iterationOutput, ...modelOutputs} = output;
    const outputNames = [...Object.values(modelOutputs), iterationOutput.status.iterationCount];
    const compiled = graph.compile(device, {
      outputs: outputNames,
      outputLocations: Object.fromEntries(outputNames.map(name => [name, 'cpu' as const]))
    });
    try {
      for (const tolerance of [0.5, 1e-5, 0.5, 0]) {
        const values = await compiled.run({
          inputs: {
            predictors: makeCPUVector('float32', Float32Array.from(seeded.predictors)),
            responses: makeCPUVector('float32', Float32Array.from(seeded.responses)),
            initial: makeCPUVector('float32', Float32Array.from(seeded.initialResiduals)),
            parameters: makeCPUVector(
              'float32',
              tolerance > 0
                ? getGPUSeeminglyUnrelatedRegressionParameterValues({tolerance})
                : Float32Array.of(0)
            )
          }
        });
        const actual = Object.fromEntries(
          outputNames.map(name => [name, getCPUVectorRows(values[name] as unknown as CPUVector)])
        ) as Record<keyof IteratedSURExpectation, Float32Array | Uint32Array>;
        if (tolerance === 0) {
          expect(Array.from(actual.status)).toEqual([5]);
          expect(Array.from(actual.iterationCount)).toEqual([0]);
          for (const name of [
            'coefficients',
            'covariance',
            'residualCovariance',
            'residuals'
          ] as const)
            expect(actual[name].every(Number.isNaN)).toBe(true);
        } else
          expectIteratedSURResult(
            actual,
            {...seeded.rounds[tolerance === 0.5 ? 1 : 2].expected, status: [0]},
            5e-4
          );
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    device.destroy();
  }
});

it('R9 CPU accepts a finite float64 tolerance above the float32 range', () => {
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(seeded.predictors),
      responses: Float32Array.from(seeded.responses),
      initialResiduals: Float32Array.from(seeded.initialResiduals)
    },
    {
      equationCount: 2,
      predictorCount: 1,
      rowCount: 8,
      iteration: {maximumIterations: 100, tolerance: 1e40}
    }
  );
  expectIteratedSURResult(result, {...seeded.rounds[1].expected, status: [0]}, 2e-6);
});

it('R9 changing the convenience tolerance does not change generated WGSL', () => {
  const sources = [0.1, 0.2].map(tolerance => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `tolerance-${serial++}`, format, length);
      const model = new GPUSeeminglyUnrelatedRegression({
        equationCount: 2,
        predictorCount: 1,
        rowCount: 8,
        predictors: view('float32', 16),
        responses: view('float32', 16),
        iteration: {maximumIterations: 2, tolerance},
        output: {
          coefficients: view('float32', 4),
          covariance: view('float32', 16),
          residualCovariance: view('float32', 4),
          residuals: view('float32', 16),
          status: view('uint32', 1)
        }
      });
      return getGPUCommandNodeKernelSources(model.getCommandNodes(graph)).map(
        kernel => kernel.source
      );
    } finally {
      device.destroy();
    }
  });
  expect(sources[0]).toEqual(sources[1]);
});

function validateRing(firstOffset: number, flag?: 'overflow' | 'candidateOverflow'): number {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `review-weights-${serial++}`, format, length);
    const offsets = view('uint32', 9);
    const neighbors = view('uint32', 8 + firstOffset);
    const weights = view('float32', 8 + firstOffset);
    const loss = view('uint32', 1);
    const output = view('float32', 1);
    const memory = createCPUGraphMemory();
    memory.write(
      offsets,
      Uint32Array.from({length: 9}, (_, row) => row + firstOffset)
    );
    memory.write(
      neighbors,
      Uint32Array.from({length: 8 + firstOffset}, (_, slot) => (slot - firstOffset + 1 + 8) % 8)
    );
    memory.write(weights, new Float32Array(8 + firstOffset).fill(1));
    memory.write(loss, Uint32Array.of(1));
    executeCommandNodesOnCPU(
      getSpatialSURWeightValidationNodes(graph, {
        id: 'review-weights',
        rowCount: 8,
        weights: {offsets, neighbors, weights, status: flag ? {[flag]: loss} : undefined},
        output
      }),
      memory
    );
    return memory.readFloat32(output)[0];
  } finally {
    device.destroy();
  }
}

for (const flag of ['overflow', 'candidateOverflow'] as const)
  it(`R3 direct GPU validator suppresses ${flag}`, () => expect(validateRing(0, flag)).toBe(1));

it('R6 rejects an unused leading CSR slot in the actual GPU validator', () =>
  expect(validateRing(1)).toBe(1));
it('R6 CPU reports invalid weights for a nonzero first CSR offset', () => {
  const weights = {
    offsets: Uint32Array.from(spatialFixture.weights.offsets, value => value + 1),
    neighbors: Uint32Array.from([0, ...spatialFixture.weights.neighbors]),
    weights: Float32Array.from([0, ...spatialFixture.weights.weights])
  };
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(spatialFixture.predictors),
      responses: Float32Array.from(spatialFixture.responses),
      spatial: {weights}
    },
    spatialFixture
  );
  expect(Array.from(result.spatial!.status)).toEqual([4, 4]);
  expect(Array.from(result.status)).toEqual([1]);
  expect(Array.from(result.iterationCount)).toEqual([0]);
  for (const name of ['coefficients', 'covariance', 'residualCovariance', 'residuals'] as const)
    expect(result[name].every(Number.isNaN)).toBe(true);
});

for (const flag of ['overflow', 'candidateOverflow']) {
  it(`R3 refuses spread spatial weight-loss status in CPU and named forms: ${flag}`, () => {
    const weights = {
      ...spatialFixture.weights,
      offsets: Uint32Array.from(spatialFixture.weights.offsets),
      neighbors: Uint32Array.from(spatialFixture.weights.neighbors),
      weights: Float32Array.from(spatialFixture.weights.weights),
      status: {[flag]: Uint32Array.of(1)}
    };
    expect(() =>
      computeSeeminglyUnrelatedRegressionOnCPU(
        {
          predictors: Float32Array.from(spatialFixture.predictors),
          responses: Float32Array.from(spatialFixture.responses),
          spatial: {weights}
        },
        spatialFixture
      )
    ).toThrow(/weights.status.*GPU contributor/);
    const namedWeights = {
      ...{offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      status: {[flag]: 'loss'}
    };
    for (const Operation of [
      GPUSeeminglyUnrelatedRegressionOperation,
      CPUSeeminglyUnrelatedRegressionOperation
    ])
      expect(
        () =>
          new Operation({
            equationCount: 2,
            predictorCount: 2,
            rowCount: 40,
            predictors: 'predictors',
            responses: 'responses',
            spatial: {
              weights: namedWeights,
              output: {parameters: 'parameters', status: 'spatialStatus'}
            },
            output: {
              coefficients: 'coefficients',
              covariance: 'covariance',
              residualCovariance: 'residualCovariance',
              residuals: 'residuals',
              status: 'status'
            }
          })
      ).toThrow(/weights.status.*GPU contributor/);
  });
}

it('R5 spatial slopes are invariant to an exactly representable response shift for accepted row sums', () => {
  const responses = Float32Array.from(spatialFixture.responses, value => Math.round(value * 8) / 8);
  const weights = {
    offsets: Uint32Array.from(spatialFixture.weights.offsets),
    neighbors: Uint32Array.from(spatialFixture.weights.neighbors),
    weights: Float32Array.from(
      spatialFixture.weights.weights,
      (value, slot) => value * (Math.floor(slot / 2) % 2 ? 1 + 8e-6 : 1 - 8e-6)
    )
  };
  const base = computeSeeminglyUnrelatedRegressionOnCPU(
    {predictors: Float32Array.from(spatialFixture.predictors), responses, spatial: {weights}},
    spatialFixture
  );
  const shifted = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(spatialFixture.predictors),
      responses: Float32Array.from(responses, value => value + 65536),
      spatial: {weights}
    },
    spatialFixture
  );
  expect(Array.from(base.status)).toEqual([0]);
  expect(Array.from(shifted.status)).toEqual([0]);
  // The spatial model contains an intercept. Adding a constant changes only that intercept.
  for (const index of [1, 2, 4, 5])
    expect(Math.abs(shifted.coefficients[index] - base.coefficients[index])).toBeLessThan(1e-6);
});

for (const shift of [0, 65536])
  it(`R5 matches independent normalized spatial GLS outputs at response offset ${shift}`, () => {
    const fixture = seeded.spatialShift;
    const result = computeSeeminglyUnrelatedRegressionOnCPU(
      {
        predictors: Float32Array.from(fixture.predictors),
        responses: Float32Array.from(fixture.responses, value => value + shift),
        spatial: {
          weights: {
            offsets: Uint32Array.from(fixture.weights.offsets),
            neighbors: Uint32Array.from(fixture.weights.neighbors),
            weights: Float32Array.from(fixture.weights.weights)
          }
        }
      },
      {equationCount: 2, predictorCount: 2, rowCount: 40}
    );
    expectIteratedSURResult(
      result,
      {
        ...fixture.expected,
        coefficients: fixture.expected.coefficients.map(
          (value, index) => value + (index % 3 === 0 ? shift : 0)
        )
      },
      2e-6
    );
    expect(Array.from(result.spatial!.status)).toEqual([0, 0]);
    fixture.expected.parameters.forEach((value, index) =>
      expect(Math.abs(result.spatial!.parameters[index] - value)).toBeLessThan(2e-6)
    );
  });

for (const maximumIterations of [undefined, 1, 2]) {
  it(`R2 rejects an exactly fitted equation before GLS: ${maximumIterations ?? 'legacy'}`, () => {
    const result = computeSeeminglyUnrelatedRegressionOnCPU(
      {
        predictors: Float32Array.from({length: 16}, (_, index) =>
          index < 8 ? index : (3 * (index - 8)) % 7
        ),
        responses: Float32Array.from({length: 16}, (_, index) =>
          index < 8 ? 1 + 2 * index : (index - 7) % 3
        )
      },
      {
        equationCount: 2,
        predictorCount: 1,
        rowCount: 8,
        iteration: maximumIterations ? {maximumIterations} : undefined
      }
    );
    expect(Array.from(result.status)).toEqual([3]);
    expect(Array.from(result.iterationCount)).toEqual([0]);
    for (const name of ['coefficients', 'covariance', 'residuals'] as const)
      expect(result[name].every(Number.isNaN), name).toBe(true);
    if (maximumIterations) expect(result.residualCovariance.every(Number.isNaN)).toBe(true);
    else {
      // Exact rational OLS on x2=[0,3,6,2,5,1,4,0], y2=[1,2,0,1,2,0,1,2]
      // gives RSS/n = 689/1148. The first equation has identically zero residual.
      expect(Array.from(result.residualCovariance.subarray(0, 3))).toEqual([0, 0, 0]);
      expect(Math.abs(result.residualCovariance[3] - 689 / 1148)).toBeLessThan(1e-7);
    }
  });
}

it('R2 preserves genuinely small variance when the response and fit are equally scaled', () => {
  const result = computeSeeminglyUnrelatedRegressionOnCPU(
    {
      predictors: Float32Array.from(seeded.predictors),
      responses: Float32Array.from(seeded.responses, value => value * 1e-12)
    },
    {equationCount: 2, predictorCount: 1, rowCount: 8}
  );
  expect(Array.from(result.status)).toEqual([0]);
  [2e-24, 1e-24, 1e-24, 2.25e-24].forEach((value, index) => {
    expect(Number.isFinite(result.residualCovariance[index])).toBe(true);
    expect(Math.abs(result.residualCovariance[index] / value - 1)).toBeLessThan(1e-5);
  });
});

for (const round of seeded.rounds) {
  it(`R1 refreshes supplied covariance before convergence, budget ${round.maximumIterations}`, () => {
    const result = computeSeeminglyUnrelatedRegressionOnCPU(
      {
        predictors: Float32Array.from(seeded.predictors),
        responses: Float32Array.from(seeded.responses),
        initialResiduals: Float32Array.from(seeded.initialResiduals)
      },
      {
        equationCount: 2,
        predictorCount: 1,
        rowCount: 8,
        iteration: {maximumIterations: round.maximumIterations}
      }
    );
    expectIteratedSURResult(result, round.expected, 2e-6);
  });
}
