// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {GPULISAMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {getGPUSpatialAutocorrelationParameterValues} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUTransitionMatrix} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix';
import {
  AUDIT_SPATIAL_VALUES,
  AUDIT_SPATIAL_CLASSES,
  AUDIT_SPATIAL_MASK,
  AUDIT_SPATIAL_WEIGHTS,
  AUDIT_SPATIAL_LAGS,
  AUDIT_SPATIAL_EXPECTED,
  AUDIT_LISA_EXPECTED
} from './audit-spatial-markov-fixture';

for (const lagIntervalClosure of ['left', 'right'] as const) {
  it(`audits all spatial Markov ${lagIntervalClosure} outputs with ties, an island and missing input`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const expected = AUDIT_SPATIAL_EXPECTED[lagIntervalClosure];
    for (const scale of [1, 2]) {
      const rig = new AnalysisRig(device);
      try {
        const output = {
          counts: rig.output('uint32', 8),
          probabilities: rig.output('float32', 8),
          rowTotals: rig.output('uint32', 4),
          ignored: rig.output('uint32', 1),
          lagBreaks: rig.output('float32', 3),
          lagClassCount: rig.output('uint32', 1),
          lagValues: rig.output('float32', 24),
          lagClasses: rig.output('uint32', 24),
          homogeneity: rig.output('float32', 5),
          steadyStates: rig.output('float32', 6),
          stateSparsity: rig.output('uint32', 2),
          status: {converged: rig.output('uint32', 1)}
        };
        rig.run(
          new GPUSpatialMarkov({
            values: rig.input(
              Float32Array.from(AUDIT_SPATIAL_VALUES, value => value * scale),
              'float32'
            ),
            classes: rig.input(Uint32Array.from(AUDIT_SPATIAL_CLASSES), 'uint32'),
            mask: rig.input(Uint32Array.from(AUDIT_SPATIAL_MASK), 'uint32'),
            weights: rig.weights(AUDIT_SPATIAL_WEIGHTS),
            periodCount: 3,
            classCount: 2,
            maximumLagClassCount: 2,
            lagIntervalClosure,
            lagParameters: rig.input(
              getGPUClassBreaksParameterValues({method: 'quantile', classCount: 2}, 2),
              'float32'
            ),
            output
          })
        );
        for (const port of [
          'counts',
          'rowTotals',
          'ignored',
          'lagClasses',
          'stateSparsity'
        ] as const) {
          expect(await rig.readUint(output[port]), port).toEqual(expected[port]);
        }
        expect(await rig.readUint(output.lagClassCount)).toEqual([2]);
        expect(await rig.readUint(output.status.converged)).toEqual(expected.converged);
        expectNumericArrayClose(
          await rig.readFloat(output.lagValues),
          AUDIT_SPATIAL_LAGS.map(value => value * scale)
        );
        expectNumericArrayClose(await rig.readFloat(output.lagBreaks), [0, 2 * scale, 10 * scale]);
        expectNumericArrayClose(await rig.readFloat(output.probabilities), expected.probabilities, {
          absoluteTolerance: 1e-7
        });
        expectNumericArrayClose(await rig.readFloat(output.steadyStates), expected.steadyStates, {
          absoluteTolerance: 2e-6
        });
        // f32 logs/sums and incomplete-gamma approximation, as documented for diagnostics.
        expectNumericArrayClose(await rig.readFloat(output.homogeneity), expected.homogeneity, {
          absoluteTolerance: (_, index) => (index === 1 || index === 3 ? 2e-4 : 0),
          relativeTolerance: 1e-5
        });
      } finally {
        rig.destroy();
      }
    }
  });
}

for (const gating of ['none', 'significance'] as const) {
  it(`audits all LISA Markov ${gating} outputs with conditional-randomization normal gating`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const expected = AUDIT_LISA_EXPECTED[gating];
    for (const affine of [false, true]) {
      const rig = new AnalysisRig(device);
      try {
        const output = {
          quadrants: rig.output('uint32', 24),
          counts: rig.output('uint32', 25),
          probabilities: rig.output('float32', 25),
          rowTotals: rig.output('uint32', 5),
          ignored: rig.output('uint32', 1),
          steadyStates: rig.output('float32', 5),
          stateSparsity: rig.output('uint32', 2),
          status: {converged: rig.output('uint32', 1)}
        };
        rig.run(
          new GPULISAMarkov({
            values: rig.input(
              Float32Array.from(AUDIT_SPATIAL_VALUES, value => (affine ? 2 * value + 3 : value)),
              'float32'
            ),
            mask: rig.input(Uint32Array.from(AUDIT_SPATIAL_MASK), 'uint32'),
            weights: rig.weights(AUDIT_SPATIAL_WEIGHTS),
            periodCount: 3,
            gating,
            parameters: rig.input(
              getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.2}),
              'float32'
            ),
            output
          })
        );
        for (const port of [
          'quadrants',
          'counts',
          'rowTotals',
          'ignored',
          'stateSparsity'
        ] as const) {
          expect(await rig.readUint(output[port]), port).toEqual(expected[port]);
        }
        expect(await rig.readUint(output.status.converged)).toEqual(expected.converged);
        expectNumericArrayClose(await rig.readFloat(output.probabilities), expected.probabilities, {
          absoluteTolerance: 1e-7
        });
        expectNumericArrayClose(await rig.readFloat(output.steadyStates), expected.steadyStates);
      } finally {
        rig.destroy();
      }
    }
  });
}

for (const failedFlag of ['overflow', 'candidateOverflow'] as const) {
  it(`suppresses every dependent spatial Markov result on weights ${failedFlag}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const output = {
        counts: rig.output('uint32', 8),
        probabilities: rig.output('float32', 8),
        rowTotals: rig.output('uint32', 4),
        ignored: rig.output('uint32', 1),
        lagBreaks: rig.output('float32', 3),
        lagClassCount: rig.output('uint32', 1),
        lagValues: rig.output('float32', 24),
        lagClasses: rig.output('uint32', 24)
      };
      rig.run(
        new GPUSpatialMarkov({
          values: rig.input(Float32Array.from(AUDIT_SPATIAL_VALUES), 'float32'),
          classes: rig.input(Uint32Array.from(AUDIT_SPATIAL_CLASSES), 'uint32'),
          mask: rig.input(Uint32Array.from(AUDIT_SPATIAL_MASK), 'uint32'),
          weights: {
            ...rig.weights(AUDIT_SPATIAL_WEIGHTS),
            status: {
              overflow: rig.input(Uint32Array.of(failedFlag === 'overflow' ? 1 : 0), 'uint32'),
              candidateOverflow: rig.input(
                Uint32Array.of(failedFlag === 'candidateOverflow' ? 1 : 0),
                'uint32'
              )
            }
          },
          periodCount: 3,
          classCount: 2,
          maximumLagClassCount: 2,
          lagParameters: rig.input(
            getGPUClassBreaksParameterValues({method: 'quantile', classCount: 2}, 2),
            'float32'
          ),
          output
        })
      );
      // Independent policy expectation: no lag class or transition can survive incomplete weights.
      expectNumericArrayClose(
        await rig.readFloat(output.lagValues),
        new Array<number>(24).fill(NaN)
      );
      expect(await rig.readUint(output.lagClassCount)).toEqual([0]);
      expect(await rig.readUint(output.lagClasses)).toEqual(new Array<number>(24).fill(0xffffffff));
      expect(await rig.readUint(output.counts)).toEqual(new Array<number>(8).fill(0));
      expectNumericArrayClose(
        await rig.readFloat(output.probabilities),
        new Array<number>(8).fill(NaN)
      );
      expect(await rig.readUint(output.rowTotals)).toEqual([0, 0, 0, 0]);
      expect(await rig.readUint(output.ignored)).toEqual([16]);
    } finally {
      rig.destroy();
    }
  });
}

it('keeps complete empty transition rows at zero probability', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const probabilities = rig.output('float32', 4);
    rig.run(
      new GPUTransitionMatrix({
        classes: rig.input(Uint32Array.of(0, 0), 'uint32'),
        classCount: 2,
        periodCount: 2,
        output: {counts: rig.output('uint32', 4), probabilities}
      })
    );
    expect(await rig.readFloat(probabilities)).toEqual([1, 0, 0, 0]);
  } finally {
    rig.destroy();
  }
});
