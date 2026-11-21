// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import type {
  CPUSpatialInteractionInput,
  CPUSpatialInteractionOptions
} from '../../../src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation';
import {GPUSpatialInteraction} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {GPUSpatialInteractionOperation} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  INPUT,
  OPTIONS,
  EXPECTED,
  EXHAUSTED,
  BOUNDED_INPUT,
  BOUNDED_OPTIONS,
  BOUNDED_EXPECTED,
  TIED_INPUT,
  TIED_OPTIONS,
  TIED_EXPECTED,
  OBSERVED_ZERO_INPUT,
  OBSERVED_ZERO_OPTIONS,
  OBSERVED_ZERO_EXPECTED,
  expectOracleResult
} from './spatial-interaction-audit-fixtures';
import {expectOracleRows} from './spatial-interaction-audit-fixtures';

for (const method of ['dense', 'sparse'] as const) {
  it(`selects the exact Poisson fit over a near fit with positive ${method} deviance`, async () => {
    const sparse = method === 'sparse';
    const coefficient = Math.log(1025 / 1024);
    const flows = sparse ? Float32Array.of(1024, 1025, 1025, 1024) : Float32Array.of(1024, 1025);
    const actual = await run(
      {
        origins: sparse ? Uint32Array.of(0, 0, 1, 1) : Uint32Array.of(0, 0),
        destinations: sparse ? Uint32Array.of(0, 1, 0, 1) : Uint32Array.of(0, 1),
        flows,
        covariate: sparse ? Float32Array.of(0, 1, 1, 0) : Float32Array.of(0, 1)
      },
      {
        originCount: sparse ? 2 : 1,
        destinationCount: 2,
        constraint: sparse ? 'doubly' : 'unconstrained',
        coefficients: [0, coefficient],
        maximumIterations: 8
      },
      method
    );
    // 2*sum(y*log(y/1024.5)-y+1024.5), independently evaluated in float64.
    expectOracleRows(
      [actual['candidates'][0]],
      [sparse ? 0.0009760859343071322 : 0.0004880429671535661],
      1e-7
    );
    expect(actual['candidates'][0]).toBeGreaterThan(0);
    expectOracleRows([actual['summary'][0]], [coefficient], 1e-8);
    expectOracleRows(actual['predictions'], Array.from(flows), 3e-4);
    expectOracleRows(actual['status'], [0], 0);
  });
}

it('retains a feasible bounded interior with a one-sweep balancing budget', async () => {
  for (const coefficients of [
    [-1, 1],
    [-1, 3]
  ]) {
    const actual = await run(
      {
        origins: Uint32Array.of(0, 0, 1, 1),
        destinations: Uint32Array.of(0, 1, 0, 1),
        flows: Float32Array.of(10, 10, 10, 10),
        covariate: Float32Array.of(0, 0, 0, 1)
      },
      {
        originCount: 2,
        destinationCount: 2,
        constraint: 'doubly',
        coefficients,
        calibration: 'bounded',
        maximumIterations: 1,
        maximumCalibrationIterations: 2,
        coefficientTolerance: 1e-5
      }
    );
    // beta=0 yields four means ten after exactly one IPF sweep. With a budget of two the
    // probe 0 is evaluated after the last refinement and must still be certified.
    expectOracleRows(actual['status'], [0], 0);
    expectOracleRows(actual['predictions'], [10, 10, 10, 10], 1e-4);
    expect(Math.abs(actual['summary'][0])).toBeLessThan(1e-5);
    expectOracleRows(actual['calibration'], [0, 0, 0, 1], 0);
  }
});

it('retains a nonzero-deviance exploratory fit without discarding unresolved coefficient ranges', async () => {
  const actual = await run(
    {
      origins: Uint32Array.of(0, 0, 1, 1),
      destinations: Uint32Array.of(0, 1, 0, 1),
      flows: Float32Array.of(10, 20, 30, 40),
      covariate: Float32Array.of(0, 0, 0, 1)
    },
    {
      originCount: 2,
      destinationCount: 2,
      constraint: 'doubly',
      coefficients: [-1, 3],
      calibration: 'bounded',
      maximumIterations: 1,
      maximumCalibrationIterations: 4,
      coefficientTolerance: 1e-5
    }
  );
  // beta=0 independent means: rowTotal*columnTotal/100.
  expectOracleRows(actual['status'], [4], 0);
  expectOracleRows(actual['predictions'], [12, 18, 28, 42], 2e-5);
  expectOracleRows(actual['calibration'], [-1, 3, 4, 0], 0);
});

it('does not certify a narrow bracket when every inner fit failed', async () => {
  const actual = await run(
    {
      origins: Uint32Array.of(0, 0, 1, 1),
      destinations: Uint32Array.of(0, 1, 0, 1),
      flows: Float32Array.of(10, 10, 10, 10),
      covariate: Float32Array.of(0, 0, 0, 1)
    },
    {
      originCount: 2,
      destinationCount: 2,
      constraint: 'doubly',
      coefficients: [1, 1 + 2 ** -20],
      calibration: 'bounded',
      maximumIterations: 1,
      maximumCalibrationIterations: 4,
      coefficientTolerance: 1e-5
    }
  );
  expectOracleRows(actual['status'], [3], 0);
  expectOracleRows([actual['calibration'][3]], [0], 0);
});

async function run(
  input: CPUSpatialInteractionInput,
  options: CPUSpatialInteractionOptions,
  method: 'dense' | 'sparse' = 'dense',
  includeCoefficientStatus = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const output = {
    predictions: 'predictions',
    summary: 'summary',
    candidates: 'candidates',
    status: 'status',
    ...(includeCoefficientStatus ? {coefficientStatus: 'coefficientStatus'} : {}),
    originResiduals: 'originResiduals',
    destinationResiduals: 'destinationResiduals',
    ...(options.calibration === 'bounded' ? {calibration: 'calibration'} : {})
  };
  return runOperationGraph(device, {
    inputs: {
      origins: {format: 'uint32', values: input.origins},
      destinations: {format: 'uint32', values: input.destinations},
      flows: {format: 'float32', values: input.flows},
      covariate: {format: 'float32', values: input.covariate},
      ...(input.pairState ? {pairState: {format: 'uint32' as const, values: input.pairState}} : {})
    },
    operations: [
      new GPUSpatialInteractionOperation({
        ...options,
        method,
        pairCount: input.flows.length,
        origins: 'origins',
        destinations: 'destinations',
        flows: 'flows',
        covariate: 'covariate',
        pairState: input.pairState ? 'pairState' : undefined,
        output
      })
    ],
    outputs: Object.values(output)
  });
}

for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const) {
  for (const method of constraint === 'doubly'
    ? (['dense', 'sparse'] as const)
    : (['dense'] as const)) {
    it(`matches every ${method}/${constraint} GPU port to the independent Poisson oracle`, async () => {
      for (const shift of [0, 8]) {
        const actual = await run(
          {...INPUT, covariate: INPUT.covariate.map(value => value + shift)},
          {...OPTIONS, constraint},
          method
        );
        expectOracleResult(actual, EXPECTED[constraint], 0, true);
      }
    });
  }
}

for (const method of ['dense', 'sparse'] as const) {
  it(`publishes every exhausted ${method} port after one IPF sweep`, async () => {
    expectOracleResult(
      await run(INPUT, {...OPTIONS, coefficients: [-Math.log(2)], maximumIterations: 1}, method),
      EXHAUSTED,
      3,
      true
    );
  });
}

it('publishes every bounded-search port after one golden-section refinement', async () => {
  expectOracleResult(await run(BOUNDED_INPUT, BOUNDED_OPTIONS), BOUNDED_EXPECTED, 4, true);
});

it('selects the first coefficient on an exact GPU deviance tie', async () => {
  expectOracleResult(await run(TIED_INPUT, TIED_OPTIONS), TIED_EXPECTED, 0, true);
});

it('keeps zero observed counts in GPU support and Poisson deviance', async () => {
  expectOracleResult(
    await run(OBSERVED_ZERO_INPUT, OBSERVED_ZERO_OPTIONS),
    OBSERVED_ZERO_EXPECTED,
    0,
    true
  );
});

for (const method of ['dense', 'sparse'] as const) {
  for (const scene of ['no-mass', 'invalid'] as const) {
    it(`publishes every ${method}/${scene} GPU output according to the support definition`, async () => {
      const flows = new Float32Array(8);
      if (scene === 'invalid') flows[0] = -1;
      expectOracleResult(
        await run({...INPUT, flows}, {...OPTIONS, coefficients: [0]}, method),
        {
          ...EXHAUSTED,
          candidates: [NaN, NaN, 0, scene === 'no-mass' ? 2 : 3]
        },
        scene === 'no-mass' ? 1 : 2,
        true
      );
    });
  }
}

it('keeps contributor inputs and budgets at their construction values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const output = {
      predictions: rig.output('float32', 8),
      summary: rig.output('float32', 4),
      candidates: rig.output('float32', 4),
      status: rig.output('uint32', 1),
      originResiduals: rig.output('float32', 3),
      destinationResiduals: rig.output('float32', 3)
    };
    const coefficients = [-Math.log(2)];
    const props = {
      ...OPTIONS,
      coefficients,
      maximumIterations: 1,
      origins: rig.input(INPUT.origins, 'uint32'),
      destinations: rig.input(INPUT.destinations, 'uint32'),
      flows: rig.input(INPUT.flows, 'float32'),
      covariate: rig.input(INPUT.covariate, 'float32'),
      pairState: rig.input(INPUT.pairState!, 'uint32'),
      output
    };
    const contributor = new GPUSpatialInteraction(props);
    props.maximumIterations = 256;
    coefficients[0] = 0;
    rig.run(contributor);
    const {status, ...floatingOutput} = output;
    const actual = {
      ...Object.fromEntries(
        await Promise.all(
          Object.entries(floatingOutput).map(async ([port, view]) => [
            port,
            await rig.readFloat(view)
          ])
        )
      ),
      status: await rig.readUint(status)
    };
    expectOracleResult(actual, EXHAUSTED, 3, true);
  } finally {
    rig.destroy();
  }
});

it('separates coefficient absorption from origin-constrained prediction success under reversed grids', async () => {
  for (const coefficients of [
    [0, 1],
    [1, 0]
  ]) {
    const actual = await run(
      {
        origins: Uint32Array.of(0, 0, 1, 1),
        destinations: Uint32Array.of(0, 1, 0, 1),
        flows: Float32Array.of(1, 3, 2, 6),
        covariate: Float32Array.of(1, 1, 7, 7)
      },
      {
        originCount: 2,
        destinationCount: 2,
        constraint: 'origin',
        coefficients
      },
      'dense',
      true
    );
    expectOracleRows(actual['status'], [0], 0);
    expectOracleRows(actual['coefficientStatus'], [1], 0);
    expectOracleRows(actual['predictions'], [2, 2, 4, 4], 5e-4);
    expectOracleRows([actual['summary'][1]], [3.13948894], 5e-4);
  }
});

it('certifies joint margin absorption on connected and disconnected supports under grid reversal', async () => {
  for (const disconnected of [false, true])
    for (const coefficients of [
      [-1, 0, 1],
      [1, 0, -1]
    ]) {
      const actual = await run(
        {
          origins: Uint32Array.from(disconnected ? [0, 0, 1, 1, 2, 2, 3, 3] : [0, 0, 1, 1]),
          destinations: Uint32Array.from(disconnected ? [0, 1, 0, 1, 2, 3, 2, 3] : [0, 1, 0, 1]),
          flows: Float32Array.from(disconnected ? [2, 3, 5, 7, 4, 6, 8, 10] : [2, 3, 5, 7]),
          covariate: Float32Array.from(disconnected ? [0, 1, 1, 2, 5, 7, 8, 10] : [0, 1, 1, 2])
        },
        {
          originCount: disconnected ? 4 : 2,
          destinationCount: disconnected ? 4 : 2,
          constraint: 'doubly',
          coefficients
        },
        'dense',
        true
      );
      expectOracleRows(actual['coefficientStatus'], [1], 0);
      expectOracleRows(
        actual['predictions'],
        disconnected
          ? [35 / 17, 50 / 17, 84 / 17, 120 / 17, 120 / 28, 160 / 28, 216 / 28, 288 / 28]
          : [35 / 17, 50 / 17, 84 / 17, 120 / 17],
        5e-4
      );
      if (!disconnected) {
        expect(Number.isFinite(actual['summary'][1])).toBe(true);
        expect(actual['summary'][1]).toBeCloseTo(0.00405469723045826, 5);
      }
    }
});

it('reports one distinct float32 coefficient as undetermined and a two-point flat profile as sampled evidence', async () => {
  for (const [coefficients, expected] of [
    [[1, 1 + 1e-8], 0],
    [[-1, 1], 2]
  ] as const) {
    const actual = await run(
      {
        origins: Uint32Array.of(0, 0),
        destinations: Uint32Array.of(0, 1),
        flows: Float32Array.of(2, 2),
        covariate: Float32Array.of(0, 1)
      },
      {originCount: 1, destinationCount: 2, constraint: 'origin', coefficients},
      'dense',
      true
    );
    expectOracleRows(actual['coefficientStatus'], [expected], 0);
  }
});

it('canonicalizes signed zeros without turning flushed subnormals into structural absorption', async () => {
  for (const [bits, expected] of [
    [[0, 0x80000000], 1],
    [[0, 1], 2]
  ] as const) {
    const actual = await run(
      {
        origins: Uint32Array.of(0, 0),
        destinations: Uint32Array.of(0, 1),
        flows: Float32Array.of(2, 3),
        covariate: new Float32Array(Uint32Array.from(bits).buffer)
      },
      {originCount: 1, destinationCount: 2, constraint: 'origin', coefficients: [-1, 1]},
      'dense',
      true
    );
    expectOracleRows(actual['coefficientStatus'], [expected], 0);
  }
});
