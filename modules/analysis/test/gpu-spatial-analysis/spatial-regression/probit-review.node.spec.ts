// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeProbitRegressionOnCPU,
  CPUProbitRegressionOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {
  GPUProbitRegression,
  type GPUProbitRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {GPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getProbitSpatialLagInput, PROBIT_SPATIAL_LAG_PROPS} from './probit-spatial-lag-test-utils';
import audit from './probit-audit-fixture.json';
import ring from './probit-review-ring-fixture.json';
import boundary from './probit-review-boundary-fixture.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

export function getRingInput(predictors: number[], response: number[]) {
  const count = response.length;
  const weights = {
    offsets: Uint32Array.from({length: count + 1}, (_, row) => row),
    neighbors: Uint32Array.from({length: count}, (_, row) => (row + 1) % count),
    weights: new Float32Array(count).fill(1)
  };
  return {
    predictors: Float32Array.from(predictors),
    response: Float32Array.from(response),
    spatialLag: {weights},
    spatialDiagnostics: {weights}
  };
}

function getContributor(
  change?: (props: GPUProbitRegressionProps, graph: GPUCommandGraph) => void
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const props: GPUProbitRegressionProps = {
    predictors: view('float32', 24),
    response: view('float32', 24),
    predictorCount: 1,
    maximumIterations: 1,
    spatialLag: {
      weights: {
        offsets: view('uint32', 25),
        neighbors: view('uint32', 24),
        weights: view('float32', 24)
      },
      output: {
        coefficients: view('float32', 3),
        standardErrors: view('float32', 3),
        status: view('uint32', 1)
      }
    },
    output: {
      coefficients: view('float32', 2),
      standardErrors: view('float32', 2),
      fitted: view('float32', 24),
      marginalEffects: view('float32', 1),
      summary: view('float32', 4),
      status: view('uint32', 1)
    }
  };
  change?.(props, graph);
  return {device, graph, props};
}

it('R1 rejects an unidentified rounding-sized rho gradient', () => {
  // Every x occurs once with each outcome: exact ordinary score at beta=0 is zero.
  for (const scale of [0.25, 1, 4]) {
    const input = getRingInput([0, 1, 3, 8, 0, 1, 3, 8], [0, 0, 0, 0, 1, 1, 1, 1]);
    input.spatialLag.weights.weights.fill(scale);
    const result = computeProbitRegressionOnCPU(input, {predictorCount: 1});
    expect(result.status[0]).toBe(0);
    expect(result.spatialLag!.status[0]).toBe(3);
    expect(result.spatialLag!.coefficients.every(Number.isNaN)).toBe(true);
    expect(result.spatialLag!.standardErrors.every(Number.isNaN)).toBe(true);
  }
});

it('R2 retains lagged intercept information and translation invariance on an induced graph', () => {
  for (const origin of [0, 5]) {
    const input = getProbitSpatialLagInput(audit);
    input.predictors = Float32Array.from(
      input.predictors,
      (value, index) => value + (index % 2 ? 0 : origin)
    );
    const result = computeProbitRegressionOnCPU(input, {predictorCount: 2, tolerance: 1e-9});
    expect(result.spatialLag!.status[0]).toBe(0);
    // Independent reviewer score-root + full [X,WX,W²X] projection, independently regenerated below.
    expect(result.spatialLag!.coefficients[3]).toBeCloseTo(-0.06931655, 5);
    expect(result.spatialLag!.standardErrors[3]).toBeCloseTo(0.70004096, 5);
    expectRegressionRowsClose(
      result.spatialLag!.coefficients,
      audit.expected.coefficients.map((value, index) =>
        index === 0 ? value - origin * audit.expected.coefficients[1] : value
      ),
      2e-5
    );
    expectRegressionRowsClose(
      result.spatialLag!.standardErrors.subarray(1),
      audit.expected.standardErrors.slice(1),
      2e-5
    );
  }
});

it('R3 centers completed instruments before testing rank on a translated regular ring', () => {
  const predictors = Array.from({length: 24}, (_, row) => ((7 * row) % 13) - 6);
  const response = predictors.map((value, row) => Number(((5 * row + 7) % 17) + 0.8 * value > 7.5));
  for (const origin of [0, 3000, 4000]) {
    const result = computeProbitRegressionOnCPU(
      getRingInput(
        predictors.map(value => value + origin),
        response
      ),
      {predictorCount: 1, tolerance: 1e-9}
    );
    expect(result.spatialLag!.status[0]).toBe(0);
    expect(result.spatialLag!.coefficients[2]).toBeCloseTo(-0.03388817, 5);
    expect(result.spatialLag!.standardErrors[2]).toBeCloseTo(0.4641739, 5);
    expectRegressionRowsClose(
      result.spatialLag!.coefficients,
      ring.expected.coefficients.map((value, index) =>
        index === 0 ? value - origin * ring.expected.coefficients[1] : value
      ),
      2e-5
    );
  }
});

it('R4 declares the named overflow status for each spatial stage independently', () => {
  const props = {
    ...PROBIT_SPATIAL_LAG_PROPS,
    spatialLag: {
      ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
      weights: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.weights,
        status: {overflow: 'lagLoss'}
      }
    },
    spatialDiagnostics: {
      ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!,
      weights: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!.weights,
        status: {overflow: 'diagnosticLoss'}
      }
    }
  };
  for (const Operation of [CPUProbitRegressionOperation, GPUProbitRegressionOperation]) {
    const operation = new Operation(props);
    expect(operation.inputs).toHaveProperty('lagWeightoverflow', 'lagLoss');
    expect(operation.inputs).toHaveProperty('diagnosticWeightoverflow', 'diagnosticLoss');
  }
});

it('R5 consumes count/requiredCount proof of loss', () => {
  for (const requiredCount of [118, 119]) {
    const input = getProbitSpatialLagInput();
    const spatial = {
      weights: {
        ...input.spatialLag.weights,
        status: {count: Uint32Array.of(118), requiredCount: Uint32Array.of(requiredCount)}
      }
    };
    const result = computeProbitRegressionOnCPU(
      {...input, spatialLag: spatial, spatialDiagnostics: spatial},
      {predictorCount: 2}
    );
    expect(result.spatialDiagnostics!.status[0]).toBe(requiredCount === 118 ? 0 : 2);
    expect(result.spatialLag!.status[0]).toBe(requiredCount === 118 ? 0 : 5);
  }
});

it('a zero completeness certificate suppresses only the probit spatial outputs', () => {
  const input = getProbitSpatialLagInput();
  for (const complete of [0, 1]) {
    const spatial = {
      weights: {...input.spatialLag.weights, status: {complete: Uint32Array.of(complete)}}
    };
    const result = computeProbitRegressionOnCPU(
      {...input, spatialLag: spatial, spatialDiagnostics: spatial},
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(0);
    expect(result.spatialDiagnostics!.status[0]).toBe(complete ? 0 : 2);
    expect(result.spatialLag!.status[0]).toBe(complete ? 0 : 5);
  }
});

it('uses the accessible CSR span when more edges are required than the terminal offset', () => {
  const input = getProbitSpatialLagInput();
  const weights = input.spatialLag.weights;
  const retained = weights.offsets[weights.offsets.length - 1];
  for (const requiredCount of [retained, retained + 1]) {
    const spatial = {
      weights: {
        ...weights,
        neighbors: Uint32Array.from([...weights.neighbors, 0]),
        weights: Float32Array.from([...weights.weights, 0]),
        status: {requiredCount: Uint32Array.of(requiredCount)}
      }
    };
    const result = computeProbitRegressionOnCPU(
      {...input, spatialLag: spatial, spatialDiagnostics: spatial},
      {predictorCount: 2}
    );
    expect(result.status[0]).toBe(0);
    expect(result.spatialDiagnostics!.status[0]).toBe(requiredCount === retained ? 0 : 2);
    expect(result.spatialLag!.status[0]).toBe(requiredCount === retained ? 0 : 5);
  }
});

it('R6 validates CSR rows before excluding their statistical contributions', () => {
  const input = getProbitSpatialLagInput(audit);
  input.spatialLag.weights.weights[input.spatialLag.weights.offsets[4]] = -1;
  const result = computeProbitRegressionOnCPU(
    {...input, spatialDiagnostics: input.spatialLag},
    {predictorCount: 2}
  );
  expect(result.status[0]).toBe(0);
  expect(result.spatialDiagnostics!.status[0]).toBe(2);
  expect(result.spatialDiagnostics!.tests.every(Number.isNaN)).toBe(true);
  expect(result.spatialLag!.status[0]).toBe(5);
});

it('R7 classifies the rounded published rho and suppresses float32 overflow', () => {
  // Full-projection oracle: rho=0.9999999954229072 rounds to exactly 1 in float32.
  // The oracle perturbs edge 16 by one ulp after rescaling the corrected full projection.
  for (const scale of [1, 2 ** -135]) {
    const input = scale === 1 ? getProbitSpatialLagInput(boundary) : getProbitSpatialLagInput();
    input.spatialLag.weights.weights = Float32Array.from(
      input.spatialLag.weights.weights,
      value => value * scale
    );
    const result = computeProbitRegressionOnCPU(input, {predictorCount: 2, tolerance: 1e-9});
    if (scale === 2 ** -135) {
      expect(result.spatialLag!.status[0]).toBe(3);
      expect(result.spatialLag!.coefficients.every(Number.isNaN)).toBe(true);
      expect(result.spatialLag!.standardErrors.every(Number.isNaN)).toBe(true);
    } else {
      expect(result.spatialLag!.coefficients[3]).toBe(1);
      expect(result.spatialLag!.status[0]).toBe(4);
    }
  }
});

it('R8 requires scalar status views in the GPU constructor too', () => {
  const fixture = getContributor((props, graph) => {
    props.spatialLag!.weights.status = {
      candidateOverflow: createTransientView(graph, 'two-words', 'uint32', 2)
    };
  });
  try {
    expect(() => new GPUProbitRegression(fixture.props)).toThrow(/candidateOverflow/);
  } finally {
    fixture.device.destroy();
  }
});

it('R8 supports an actual empty CSR in the GPU form', () => {
  const fixture = getContributor();
  fixture.props.spatialLag!.weights.neighbors = createTransientView(
    fixture.graph,
    'empty-neighbors',
    'uint32',
    0
  );
  fixture.props.spatialLag!.weights.weights = createTransientView(
    fixture.graph,
    'empty-weights',
    'float32',
    0
  );
  try {
    expect(() =>
      new GPUProbitRegression(fixture.props).getCommandNodes(fixture.graph)
    ).not.toThrow();
  } finally {
    fixture.device.destroy();
  }
});

it('R9 refuses aliased ordinary and lag output statuses', () => {
  const fixture = getContributor(props => {
    props.spatialLag!.output.status = props.output.status;
  });
  try {
    expect(() => new GPUProbitRegression(fixture.props)).toThrow(/overlap|disjoint|alias/);
  } finally {
    fixture.device.destroy();
  }
});

it('R10 protects the validated public snapshot against mutation', () => {
  const fixture = getContributor();
  try {
    const contributor = new GPUProbitRegression(fixture.props);
    try {
      contributor.props.maximumIterations = 4;
    } catch {
      /* Frozen snapshots may throw. */
    }
    expect(contributor.getCommandNodes(fixture.graph).length).toBe(
      contributor.estimatedCommandNodeCount
    );
    expect(Object.isFrozen(contributor.props.spatialLag!.output)).toBe(true);
  } finally {
    fixture.device.destroy();
  }
});

it('R5 R6 R8 execute named CPU validation and loss decisions on the same layouts', async () => {
  for (const scene of ['complete', 'loss', 'capacity', 'masked', 'twoWords', 'empty'] as const) {
    const data = scene === 'masked' ? audit : scene === 'empty' ? ring : undefined;
    const input = getProbitSpatialLagInput(data);
    if (scene === 'masked')
      input.spatialLag.weights.weights[input.spatialLag.weights.offsets[4]] = -1;
    if (scene === 'empty') {
      input.spatialLag.weights.offsets.fill(0);
      input.spatialLag.weights.neighbors = new Uint32Array();
      input.spatialLag.weights.weights = new Float32Array();
    }
    const status =
      scene === 'twoWords'
        ? {candidateOverflow: 'flag'}
        : scene === 'capacity'
          ? {requiredCount: 'required'}
          : scene === 'complete' || scene === 'loss'
            ? {count: 'written', requiredCount: 'required'}
            : undefined;
    const operation = new CPUProbitRegressionOperation({
      ...PROBIT_SPATIAL_LAG_PROPS,
      rowCount: input.response.length,
      predictorCount: scene === 'empty' ? 1 : 2,
      spatialDiagnostics: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!,
        weights: {...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!.weights, status}
      },
      spatialLag: {
        ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
        weights: {...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.weights, status}
      }
    });
    const values: Record<string, Float32Array | Uint32Array> = {
      predictors: input.predictors,
      response: input.response,
      mask: input.mask,
      ...input.spatialLag.weights,
      written: Uint32Array.of(118),
      required: Uint32Array.of(scene === 'complete' ? 118 : 119),
      flag: Uint32Array.of(0, 1)
    };
    const inputs = Object.fromEntries(
      Object.entries(operation.inputs).map(([port, name]): [string, GPUCPUVector] => {
        const rows = values[name];
        return [
          port,
          {
            format: rows instanceof Uint32Array ? 'uint32' : 'float32',
            length: rows.length,
            chunks: [{values: rows, length: rows.length}],
            pack: () => rows
          }
        ];
      })
    );
    if (scene === 'twoWords') {
      expect(() => operation.execute({inputs})).toThrow(/candidateOverflow/);
      continue;
    }
    const result = await operation.execute({inputs});
    expect((result['status'] as Uint32Array)[0]).toBe(0);
    expect((result['diagnosticStatus'] as Uint32Array)[0]).toBe(
      scene === 'complete' ? 0 : scene === 'empty' ? 3 : 2
    );
    expect((result['lagStatus'] as Uint32Array)[0]).toBe(
      scene === 'complete' ? 0 : scene === 'empty' ? 2 : 5
    );
    if (scene !== 'complete')
      for (const port of ['tests', 'lagCoefficients', 'lagStandardErrors'])
        expect((result[port] as Float32Array).every(Number.isNaN)).toBe(true);
  }
});
