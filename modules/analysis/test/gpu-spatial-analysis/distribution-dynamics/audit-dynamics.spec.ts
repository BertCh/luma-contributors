// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUClassAssignment} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {GPUClassificationFit} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit';
import {GPUTransitionMatrix} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix';
import {GPUTransitionDiagnostics} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {GPUTransitionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics-operation';
import {CPUTransitionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  AUDIT_CONDITIONS,
  AUDIT_FIT_CLASSES,
  AUDIT_FIT_EXPECTED,
  AUDIT_FIT_MASK,
  AUDIT_FIT_VALUES,
  AUDIT_PANEL,
  AUDIT_PANEL_MASK,
  AUDIT_PARALLEL_DIAGNOSTICS,
  AUDIT_SUPPORT_CASES,
  AUDIT_TRANSITIONS
} from './audit-dynamics-fixture';

it('audits class boundaries under an increasing affine transformation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Hand derivation: count inner edges <= x (left), or < x (right).
  // Includes tied edge values, a masked row, NaN and clamped infinities.
  const values = [-Infinity, -1, 0, 1, 1, 2, 3, Infinity, NaN, 2];
  const mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
  const noClass = 0xffffffff;
  for (const affine of [false, true]) {
    for (const intervalClosure of ['left', 'right'] as const) {
      const rig = new AnalysisRig(device);
      try {
        const output = rig.output('uint32', values.length);
        const transform = (value: number) => (affine ? 2 * value + 3 : value);
        rig.run(
          new GPUClassAssignment({
            values: rig.input(Float32Array.from(values, transform), 'float32'),
            breaks: rig.input(Float32Array.from([0, 1, 2, 3], transform), 'float32'),
            classCount: rig.input(Uint32Array.of(3), 'uint32'),
            mask: rig.input(Uint32Array.from(mask), 'uint32'),
            intervalClosure,
            output
          })
        );
        expect(await rig.readUint(output)).toEqual(
          intervalClosure === 'left'
            ? [0, 0, 0, 1, 1, 2, 2, 2, noClass, noClass]
            : [0, 0, 0, 0, 0, 1, 2, 2, noClass, noClass]
        );
      } finally {
        rig.destroy();
      }
    }
  }
});

it('audits every classification fit output against NumPy literals including affine equivariance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [fixtureIndex, expected] of AUDIT_FIT_EXPECTED.entries()) {
    const rig = new AnalysisRig(device);
    try {
      const output = {
        counts: rig.output('uint32', 5),
        medians: rig.output('float32', 5),
        absoluteDeviations: rig.output('float32', 5),
        squaredDeviations: rig.output('float32', 5),
        summary: rig.output('float32', 5)
      };
      rig.run(
        new GPUClassificationFit({
          values: rig.input(
            Float32Array.from(AUDIT_FIT_VALUES, value =>
              fixtureIndex === 0 ? value : 2 * value + 3
            ),
            'float32'
          ),
          classes: rig.input(Uint32Array.from(AUDIT_FIT_CLASSES), 'uint32'),
          mask: rig.input(Uint32Array.from(AUDIT_FIT_MASK), 'uint32'),
          classCount: 4,
          output
        })
      );
      expect(await rig.readUint(output.counts)).toEqual(expected.counts);
      for (const [port, values] of [
        [output.medians, expected.medians],
        [output.absoluteDeviations, expected.absolute],
        [output.squaredDeviations, expected.squared],
        [output.summary, expected.summary]
      ] as const) {
        // Small integer fixtures have exact sums; 2e-6 allows float32 quotients in the summary.
        expectNumericArrayClose(await rig.readFloat(port), values, {absoluteTolerance: 2e-6});
      }
    } finally {
      rig.destroy();
    }
  }
});

it('retains tiny classification fit for a single outlier among four million zeros', async () => {
  // Exact closed forms for float32(-1.1): SS=o^2*(1-1/N), GVF=1/(N-1).
  // One outlier plus zero medians avoids serially summing millions of copies of 1.1.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const length = 2 ** 22;
    const values = new Float32Array(length);
    values[0] = -1.1;
    const classes = new Uint32Array(length);
    classes.fill(1, length / 2);
    const output = {
      counts: rig.output('uint32', 3),
      medians: rig.output('float32', 3),
      absoluteDeviations: rig.output('float32', 3),
      squaredDeviations: rig.output('float32', 3),
      summary: rig.output('float32', 5)
    };
    rig.run(
      new GPUClassificationFit({
        values: rig.input(values, 'float32'),
        classes: rig.input(classes, 'uint32'),
        classCount: 2,
        output
      })
    );
    expectNumericArrayClose(await rig.readUint(output.counts), [2097152, 2097152, 4194304]);
    expectNumericArrayClose(await rig.readFloat(output.medians), [0, 0, 0]);
    expectNumericArrayClose(
      await rig.readFloat(output.absoluteDeviations),
      [1.100000023841858, 0, 1.100000023841858]
    );
    expectNumericArrayClose(
      await rig.readFloat(output.squaredDeviations),
      [1.2099994754791015, 0, 1.2099997639655948],
      {absoluteTolerance: 2e-7}
    );
    const summary = await rig.readFloat(output.summary);
    // Float32 division/subtraction limits tiny GVF to roughly one ulp at one; zero still fails.
    expectNumericArrayClose([summary[3]], [2.384186359449949e-7], {absoluteTolerance: 8e-8});
    expectNumericArrayClose(
      [summary[0], summary[1], summary[2], summary[4]],
      [0, 1.100000023841858, 1.100000023841858, 1.2099994754791015],
      {absoluteTolerance: 2e-7}
    );
  } finally {
    rig.destroy();
  }
}, 120_000);

it('retains classification squared deviations and GVF under a large exactly representable translation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const offset of [0, 1000000]) {
    const rig = new AnalysisRig(device);
    try {
      const output = {
        counts: rig.output('uint32', 3),
        medians: rig.output('float32', 3),
        absoluteDeviations: rig.output('float32', 3),
        squaredDeviations: rig.output('float32', 3),
        summary: rig.output('float32', 5)
      };
      rig.run(
        new GPUClassificationFit({
          values: rig.input(
            Float32Array.from(
              [0, 0, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
              value => value + offset
            ),
            'float32'
          ),
          classes: rig.input(Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1), 'uint32'),
          classCount: 2,
          output
        })
      );
      expectNumericArrayClose(await rig.readUint(output.counts), [4, 4, 8]);
      expectNumericArrayClose(
        await rig.readFloat(output.medians),
        Float32Array.from([offset + 0.03125, offset + 0.0625, offset + 0.0625])
      );
      expectNumericArrayClose(await rig.readFloat(output.absoluteDeviations), [0.125, 0, 0.125]);
      expectNumericArrayClose(
        await rig.readFloat(output.squaredDeviations),
        [0.00390625, 0, 0.005859375],
        {relativeTolerance: 2e-6}
      );
      expectNumericArrayClose(
        await rig.readFloat(output.summary),
        [0, 0.125, 0.125, 1 / 3, 0.00390625],
        {absoluteTolerance: 1e-7}
      );
    } finally {
      rig.destroy();
    }
  }
});

it('audits every conditioned transition and diagnostic output against SciPy at two period lags', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const expected of AUDIT_TRANSITIONS) {
    const rig = new AnalysisRig(device);
    try {
      const output = {
        counts: rig.output('uint32', 18),
        probabilities: rig.output('float32', 18),
        rowTotals: rig.output('uint32', 6),
        ignored: rig.output('uint32', 1),
        homogeneity: rig.output('float32', 5),
        steadyStates: rig.output('float32', 9),
        stateSparsity: rig.output('uint32', 2),
        status: {converged: rig.output('uint32', 1)}
      };
      rig.run(
        new GPUTransitionMatrix({
          classes: rig.input(Uint32Array.from(AUDIT_PANEL), 'uint32'),
          conditionClasses: rig.input(Uint32Array.from(AUDIT_CONDITIONS), 'uint32'),
          mask: rig.input(Uint32Array.from(AUDIT_PANEL_MASK), 'uint32'),
          classCount: 3,
          conditionCount: 2,
          periodCount: 4,
          periodLag: expected.periodLag,
          output
        })
      );
      for (const port of ['counts', 'rowTotals', 'ignored', 'stateSparsity'] as const) {
        expect(await rig.readUint(output[port]), port).toEqual(expected[port]);
      }
      expect(await rig.readUint(output.status.converged)).toEqual(expected.converged);
      expectNumericArrayClose(await rig.readFloat(output.probabilities), expected.probabilities, {
        absoluteTolerance: 1e-7
      });
      expectNumericArrayClose(await rig.readFloat(output.steadyStates), expected.steadyStates, {
        absoluteTolerance: 2e-6
      });
      // Float32 accumulation/logs: statistics 1e-5 relative; incomplete-gamma tails 2e-4 absolute.
      expectNumericArrayClose(await rig.readFloat(output.homogeneity), expected.homogeneity, {
        absoluteTolerance: (_, index) => (index === 1 || index === 3 ? 2e-4 : 0),
        relativeTolerance: 1e-5
      });
    } finally {
      rig.destroy();
    }
  }
});

it('audits standalone diagnostics for reducible, periodic and unused-state chains', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const expected of AUDIT_SUPPORT_CASES) {
    const rig = new AnalysisRig(device);
    try {
      const output = {
        steadyStates: rig.output('float32', 2),
        stateSparsity: rig.output('uint32', 2),
        status: {converged: rig.output('uint32', 1)}
      };
      rig.run(
        new GPUTransitionDiagnostics({
          counts: rig.input(Uint32Array.from(expected.counts), 'uint32'),
          classCount: 2,
          output
        })
      );
      expectNumericArrayClose(await rig.readFloat(output.steadyStates), expected.steadyStates);
      expect(await rig.readUint(output.stateSparsity)).toEqual(expected.stateSparsity);
      expect(await rig.readUint(output.status.converged)).toEqual(expected.converged);
    } finally {
      rig.destroy();
    }
  }
});

it('audits both diagnostic operation forms against the same independent literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const expected of AUDIT_TRANSITIONS) {
    for (const Operation of [
      GPUTransitionDiagnosticsOperation,
      CPUTransitionDiagnosticsOperation
    ]) {
      const actual = await runOperationGraph(device, {
        inputs: {counts: {format: 'uint32', values: Uint32Array.from(expected.counts)}},
        operations: [
          new Operation({
            counts: 'counts',
            classCount: 3,
            conditionCount: 2,
            output: {
              homogeneity: 'homogeneity',
              steadyStates: 'steadyStates',
              stateSparsity: 'stateSparsity',
              status: {converged: 'converged'}
            }
          })
        ],
        outputs: ['homogeneity', 'steadyStates', 'stateSparsity', 'converged']
      });
      expectNumericArrayClose(actual['homogeneity'], expected.homogeneity, {
        absoluteTolerance: (_, index) => (index === 1 || index === 3 ? 2e-4 : 0),
        relativeTolerance: 1e-5
      });
      expectNumericArrayClose(actual['steadyStates'], expected.steadyStates, {
        absoluteTolerance: 2e-6
      });
      expectNumericArrayClose(actual['stateSparsity'], expected.stateSparsity);
      expectNumericArrayClose(actual['converged'], expected.converged);
    }
  }
});

it('audits every diagnostic port across the public reduction workgroup boundary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    // Independent SciPy fixture: two different transition laws, reducible/empty chains and
    // an unused state. 257 conditions exercise the tail of both public reduction layouts.
    const expected = AUDIT_PARALLEL_DIAGNOSTICS;
    const output = {
      homogeneity: rig.output('float32', 5),
      stateSparsity: rig.output('uint32', 2),
      steadyStates: rig.output('float32', 258 * 3),
      status: {converged: rig.output('uint32', 1)}
    };
    rig.run(
      new GPUTransitionDiagnostics({
        counts: rig.input(Uint32Array.from(expected.counts), 'uint32'),
        classCount: 3,
        conditionCount: 257,
        output
      })
    );
    const homogeneity = await rig.readFloat(output.homogeneity);
    for (const [index, value] of expected.homogeneity.entries()) {
      // Accumulation is f32; the tiny Q tail also includes log-gamma cancellation at df=510.
      // Pure relative tolerance rejects a false zero for its representable 1.65e-29 value.
      // The LR tail is below the smallest float32, so it must be exactly zero.
      expectNumericArrayClose([homogeneity[index]], [value], {
        absoluteTolerance: index === 1 || index === 3 ? 0 : 1e-5,
        relativeTolerance: index === 1 ? 2e-3 : 1e-5
      });
    }
    expectNumericArrayClose(await rig.readUint(output.stateSparsity), expected.stateSparsity);
    expectNumericArrayClose(await rig.readFloat(output.steadyStates), expected.steadyStates, {
      absoluteTolerance: 2e-6
    });
    expectNumericArrayClose(await rig.readUint(output.status.converged), expected.converged);
  } finally {
    rig.destroy();
  }
});

it('reuses the public pooled counts when only conditional steady states are requested', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const expected = AUDIT_PARALLEL_DIAGNOSTICS;
    const output = {
      steadyStates: rig.output('float32', 258 * 3),
      status: {converged: rig.output('uint32', 1)}
    };
    rig.run(
      new GPUTransitionDiagnostics({
        counts: rig.input(Uint32Array.from(expected.counts), 'uint32'),
        classCount: 3,
        conditionCount: 257,
        output
      })
    );
    expectNumericArrayClose(await rig.readFloat(output.steadyStates), expected.steadyStates, {
      absoluteTolerance: 2e-6
    });
    expectNumericArrayClose(await rig.readUint(output.status.converged), expected.converged);
  } finally {
    rig.destroy();
  }
});

it('retains a sub-ulp expected-count difference in homogeneity inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const homogeneity = rig.output('float32', 5);
    rig.run(
      new GPUTransitionDiagnostics({
        counts: rig.input(Uint32Array.of(4721, 4722, 0, 0, 4720, 4721, 0, 0), 'uint32'),
        classCount: 2,
        conditionCount: 2,
        output: {homogeneity}
      })
    );
    const actual = await rig.readFloat(homogeneity);
    expectNumericArrayClose([actual[2]], [2.375953939555561e-12], {relativeTolerance: 2e-5});
    expectNumericArrayClose([actual[3]], [0.9999987701303906], {absoluteTolerance: 1e-7});
    expect(actual[4]).toBe(1);
  } finally {
    rig.destroy();
  }
});

for (const [first, second, likelihood, probability] of [
  [4143, 4141, 0.000965717054287385, 0.9752089389095286],
  [574, 528, 3.8414063913019163, 0.05000156344204883]
]) {
  it(`preserves small LR and its separate tail for nearby transition regimes ${first}/${second}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const homogeneity = rig.output('float32', 5);
      rig.run(
        new GPUTransitionDiagnostics({
          counts: rig.input(Uint32Array.of(first, second, 0, 0, second, first, 0, 0), 'uint32'),
          classCount: 2,
          conditionCount: 2,
          output: {homogeneity}
        })
      );
      const actual = await rig.readFloat(homogeneity);
      expectNumericArrayClose([actual[2]], [likelihood], {relativeTolerance: 2e-5});
      expectNumericArrayClose([actual[3]], [probability], {absoluteTolerance: 1e-6});
      expect(actual[3]).toBeGreaterThan(0.05);
    } finally {
      rig.destroy();
    }
  });
}

it('keeps large aggregated counts from wrapping into empty states', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const output = {
      homogeneity: rig.output('float32', 5),
      steadyStates: rig.output('float32', 6),
      stateSparsity: rig.output('uint32', 2),
      status: {converged: rig.output('uint32', 1)}
    };
    // Identical symmetric chains: pi=[1/2,1/2], Q=LR=0, df=2, tails=1.
    // Each row totals 2^32; pooling doubles it. uint32 aggregation used to wrap to zero.
    rig.run(
      new GPUTransitionDiagnostics({
        counts: rig.input(
          Uint32Array.of(0xffffffff, 1, 1, 0xffffffff, 0xffffffff, 1, 1, 0xffffffff),
          'uint32'
        ),
        classCount: 2,
        conditionCount: 2,
        output
      })
    );
    expectNumericArrayClose(await rig.readFloat(output.homogeneity), [0, 1, 0, 1, 2]);
    expectNumericArrayClose(
      await rig.readFloat(output.steadyStates),
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
      {absoluteTolerance: 2e-6}
    );
    expect(await rig.readUint(output.stateSparsity)).toEqual([0, 0]);
    expect(await rig.readUint(output.status.converged)).toEqual([1]);
  } finally {
    rig.destroy();
  }
});

it('clears convergence when float32 stationary weights overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const counts = new Uint32Array(8 * 8);
    for (let state = 0; state < 8; state++) {
      if (state < 7) counts[state * 8 + state + 1] = 1_000_000_000;
      if (state > 0) counts[state * 8 + state - 1] = 1;
    }
    counts[63] = 1_000_000_000;
    const output = {
      steadyStates: rig.output('float32', 8),
      stateSparsity: rig.output('uint32', 2),
      status: {converged: rig.output('uint32', 1)}
    };
    // Detailed balance implies successive stationary weight ratios of about 10^9.
    // Unscaled GTH weights exceed float32 after five states (10^45 > 3.4e38).
    // This documented numeric refusal must clear status, rather than certify NaNs.
    rig.run(
      new GPUTransitionDiagnostics({counts: rig.input(counts, 'uint32'), classCount: 8, output})
    );
    expectNumericArrayClose(await rig.readFloat(output.steadyStates), [
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      NaN,
      NaN
    ]);
    expect(await rig.readUint(output.stateSparsity)).toEqual([0, 0]);
    expect(await rig.readUint(output.status.converged)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

for (const [count, likelihood] of [
  [2 ** 27, 1.0339757656912846e-25],
  [2 ** 23, 4.235164736271532e-22]
]) {
  it(`retains the integer LR residual and separate tail for GPU counts near ${count}`, async () => {
    // Decimal(110): 2*sum(n*ln(n*N/(rowTotal*columnTotal))), [[k,k+1],[k-1,k]].
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const homogeneity = rig.output('float32', 5);
      rig.run(
        new GPUTransitionDiagnostics({
          counts: rig.input(
            Uint32Array.of(count, count + 1, 0, 0, count - 1, count, 0, 0),
            'uint32'
          ),
          classCount: 2,
          conditionCount: 2,
          output: {homogeneity}
        })
      );
      const actual = await rig.readFloat(homogeneity);
      expect(actual[2]).toBeGreaterThan(0);
      expectNumericArrayClose([actual[2]], [likelihood], {relativeTolerance: 2e-5});
      // df=1; sf is independently 0.9999999999997434 at k=2^27, rounding to one in f32.
      expect(actual[3]).toBe(1);
      expect(actual[4]).toBe(1);
    } finally {
      rig.destroy();
    }
  });
}
