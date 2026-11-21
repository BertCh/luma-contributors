// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalMoran,
  GPUHotSpotAnalysis,
  GPULocalGeary,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  AUDIT_VALUES,
  AUDIT_MASK,
  AUDIT_WEIGHTS,
  AUDIT_REFERENCE,
  AUDIT_NEIGHBOR_COUNTS,
  AUDIT_QUADRANTS,
  AUDIT_GI_BINS,
  AUDIT_UNGATED_QUADRANTS
} from './local-autocorrelation-audit-fixture';
import {FIXED_POPULATION_REFERENCE} from './local-autocorrelation-audit-fixture';

it('fixed moments decode exact extended counts on a small resident subset for all three contributors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of FIXED_POPULATION_REFERENCE) {
    for (const model of ['moran', 'gi', 'star', 'geary'] as const) {
      const rig = new WeightsRig(device);
      const result = rig.output('float32', 10);
      const localI = rig.output('float32', 10);
      const globalStatistics = rig.output('float32', 5);
      const globalCounts = rig.output('uint32', 2);
      const props = {
        weights: rig.uploadWeights(AUDIT_WEIGHTS),
        values: rig.input(Float32Array.from(AUDIT_VALUES), 'float32', 10),
        mask: rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10),
        parameters: rig.input(
          getGPUSpatialAutocorrelationParameterValues({
            fixedMoments: {count: reference.count, mean: 4, variance: 25}
          }),
          'float32',
          8
        ),
        islands: 'keep' as const,
        output: {globalStatistics: globalStatistics.view, globalCounts: globalCounts.view}
      };
      try {
        rig.run(
          model === 'moran'
            ? new GPULocalMoran({
                ...props,
                output: {...props.output, zScores: result.view, localI: localI.view}
              })
            : model === 'geary'
              ? new GPULocalGeary({...props, output: {...props.output, statistic: result.view}})
              : new GPUHotSpotAnalysis({
                  ...props,
                  selfWeight: model === 'gi' ? 0 : 1,
                  output: {...props.output, zScores: result.view}
                })
        );
        compare(
          await readFloat32(result.buffer, 10),
          model === 'geary'
            ? [0.08, 0.36, 0.36, 1.24, 1.08, 1.08, 11.88, NaN, 0, NaN]
            : reference[model],
          model
        );
        if (model === 'moran')
          compare(await readFloat32(localI.buffer, 10), reference.localI, 'fixed localI');
        compare(
          await readFloat32(globalStatistics.buffer, 5),
          [reference.encoded, 4, 25, 5, 1],
          'fixed summary'
        );
        expect(await readUint32(globalCounts.buffer, 2)).toEqual([reference.count, 1]);
      } finally {
        rig.destroy();
      }
    }
  }
});

function compare(actual: readonly number[], expected: readonly number[], label: string) {
  expect(actual.length, label).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(actual[index], `${label}[${index}]`).toBeNaN();
    } else {
      expect(Number.isFinite(actual[index]), `${label}[${index}]`).toBe(true);
      // Ten rows, short weighted sums and two-pass moments: allow 2e-5 absolute + 3e-6 relative
      // for f32 reductions, divisions, sqrt and the erfc approximation, never a sign/gate skip.
      expect(Math.abs(actual[index] - value), `${label}[${index}]`).toBeLessThanOrEqual(
        2e-5 + 3e-6 * Math.abs(value)
      );
    }
  }
}

for (const model of ['moran', 'gi', 'star'] as const) {
  it(`${model} publishes every output against exhaustive null distributions and affine copies`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const reference of AUDIT_REFERENCE) {
      const rig = new WeightsRig(device);
      const values = rig.input(Float32Array.from(AUDIT_VALUES), 'float32', 10);
      const weightOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weightCandidateOverflow = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weights = {
        ...rig.uploadWeights(AUDIT_WEIGHTS),
        information: reference.rowStandardized
          ? {rowCount: 10, transform: 'row' as const, symmetric: 'unknown' as const}
          : undefined,
        status: {overflow: weightOverflow, candidateOverflow: weightCandidateOverflow}
      };
      const zScores = rig.output('float32', 10);
      const pValues = rig.output('float32', 10);
      const adjustedPValues = rig.output('float32', 10);
      const neighborCounts = rig.output('uint32', 10);
      const globalStatistics = rig.output('float32', 5);
      const globalCounts = rig.output('uint32', 2);
      const invalidCount = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      const candidateOverflow = rig.output('uint32', 1);
      const localI = model === 'moran' ? rig.output('float32', 10) : undefined;
      const spatialLag = model === 'moran' ? rig.output('float32', 10) : undefined;
      const quadrants = model === 'moran' ? rig.output('uint32', 10) : undefined;
      const bins = model !== 'moran' ? rig.output('sint32', 10) : undefined;
      const common = {
        weights,
        values,
        mask: rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10),
        parameters: rig.input(
          getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.2}),
          'float32',
          8
        ),
        islands: reference.islands,
        output: {
          zScores: zScores.view,
          pValues: pValues.view,
          adjustedPValues: adjustedPValues.view,
          neighborCounts: neighborCounts.view,
          globalStatistics: globalStatistics.view,
          globalCounts: globalCounts.view,
          status: {
            invalidCount: invalidCount.view,
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view
          }
        }
      };
      rig.graph.add(
        model === 'moran'
          ? new GPULocalMoran({
              ...common,
              output: {
                ...common.output,
                localI: localI!.view,
                spatialLag: spatialLag!.view,
                quadrants: quadrants!.view
              }
            })
          : new GPUHotSpotAnalysis({
              ...common,
              selfWeight: model === 'gi' ? 0 : 1,
              output: {...common.output, bins: bins!.view}
            })
      );
      const compiled = rig.graph.compile();
      try {
        for (const scale of [1, 2]) {
          rig
            .bufferOf(values)
            .write(
              Float32Array.from(AUDIT_VALUES, value => value * scale + (scale === 2 ? 32 : 0))
            );
          submitGraph(device, compiled, undefined);
          const expectedScores =
            model === 'moran' ? reference.moranZ : model === 'gi' ? reference.giZ : reference.starZ;
          const expectedPValues =
            model === 'moran' ? reference.moranP : model === 'gi' ? reference.giP : reference.starP;
          compare(await readFloat32(zScores.buffer, 10), expectedScores, 'zScores');
          compare(await readFloat32(pValues.buffer, 10), expectedPValues, 'pValues');
          compare(
            await readFloat32(adjustedPValues.buffer, 10),
            expectedPValues,
            'adjustedPValues'
          );
          expect(await readUint32(neighborCounts.buffer, 10)).toEqual(AUDIT_NEIGHBOR_COUNTS);
          expect(await readUint32(globalCounts.buffer, 2)).toEqual(reference.globalCounts);
          compare(
            await readFloat32(globalStatistics.buffer, 5),
            [
              reference.globalStatistics[0],
              reference.globalStatistics[1] * scale + (scale === 2 ? 32 : 0),
              reference.globalStatistics[2] * scale * scale,
              reference.globalStatistics[3] * scale,
              1
            ],
            'globalStatistics'
          );
          expect(await readUint32(invalidCount.buffer, 1)).toEqual([1]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
          expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
          if (localI && spatialLag && quadrants) {
            compare(await readFloat32(localI.buffer, 10), reference.localI, 'localI');
            compare(
              await readFloat32(spatialLag.buffer, 10),
              reference.spatialLag.map(value => value * scale),
              'spatialLag'
            );
            expect(await readUint32(quadrants.buffer, 10)).toEqual(AUDIT_QUADRANTS);
          }
          if (bins) {
            const bits = Uint32Array.from(await readUint32(bins.buffer, 10));
            expect(Array.from(new Int32Array(bits.buffer))).toEqual(
              model === 'gi' ? AUDIT_GI_BINS : reference.starBins
            );
          }
        }
        // Every result is suppressed for either producer-loss flag. Previously computed outputs
        // must be overwritten; counts still describe selected rows and the one-pass island pass.
        for (const loss of ['overflow', 'candidateOverflow'] as const) {
          rig.bufferOf(weightOverflow).write(Uint32Array.of(loss === 'overflow' ? 7 : 0));
          rig
            .bufferOf(weightCandidateOverflow)
            .write(Uint32Array.of(loss === 'candidateOverflow' ? 9 : 0));
          submitGraph(device, compiled, undefined);
          for (const output of [zScores, pValues, adjustedPValues, localI, spatialLag]) {
            if (output)
              expect((await readFloat32(output.buffer, 10)).every(Number.isNaN)).toBe(true);
          }
          expect(await readUint32(neighborCounts.buffer, 10)).toEqual(new Array(10).fill(0));
          if (quadrants)
            expect(await readUint32(quadrants.buffer, 10)).toEqual(new Array(10).fill(0));
          if (bins) expect(await readUint32(bins.buffer, 10)).toEqual(new Array(10).fill(0));
          expect(await readUint32(invalidCount.buffer, 1)).toEqual([1]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([Number(loss === 'overflow')]);
          expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([
            Number(loss === 'candidateOverflow')
          ]);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  });
}

it('Moran ungated and Bonferroni epilogues use the same independently checked statistic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const ungated of [false, true]) {
    const rig = new WeightsRig(device);
    const zScores = rig.output('float32', 10);
    const adjusted = rig.output('float32', 10);
    const quadrants = rig.output('uint32', 10);
    try {
      rig.run(
        new GPULocalMoran({
          weights: rig.uploadWeights(AUDIT_WEIGHTS),
          values: rig.input(Float32Array.from(AUDIT_VALUES), 'float32', 10),
          mask: rig.input(Uint32Array.from(AUDIT_MASK), 'uint32', 10),
          parameters: rig.input(
            getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.2}),
            'float32',
            8
          ),
          quadrantGating: ungated ? 'none' : 'analytic',
          multipleTesting: ungated ? 'none' : 'bonferroni',
          output: {zScores: zScores.view, adjustedPValues: adjusted.view, quadrants: quadrants.view}
        })
      );
      compare(await readFloat32(zScores.buffer, 10), AUDIT_REFERENCE[0].moranZ, 'zScores');
      // Exactly seven defined tests; Bonferroni p'=min(1,7p), all exceed alpha=0.2.
      compare(
        await readFloat32(adjusted.buffer, 10),
        AUDIT_REFERENCE[0].moranP.map(value => (ungated ? value : Math.min(1, 7 * value))),
        'adjusted'
      );
      expect(await readUint32(quadrants.buffer, 10)).toEqual(
        ungated ? AUDIT_UNGATED_QUADRANTS : new Array(10).fill(0)
      );
    } finally {
      rig.destroy();
    }
  }
});
