// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import {CPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {GPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {CPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  AUDIT_CHANGE_CASES,
  AUDIT_CHANGE_SERIES,
  AUDIT_CHANGE_AFFINE_COSTS,
  AUDIT_PETTITT_EXPECTED,
  AUDIT_PETTITT_SERIES,
  REVIEW_TWO_CHANGE_CASES,
  WIDE_MEAN_COST_EXPECTED,
  WIDE_MEAN_COST_VALUES,
  WIDE_MEAN_SQUARES_EXPECTED,
  WIDE_MEAN_SQUARES_VALUES,
  WIDE_STANDARDIZATION_CASES,
  packAuditSeries
} from './audit-change-points-fixture';

it('W5-21 preserves finite GPU mean costs when squaring the total would overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const actual = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: WIDE_MEAN_COST_VALUES},
      parameters: {format: 'float32', values: Float32Array.of(1, 8)}
    },
    operations: [
      new GPULocationChangePointsOperation({
        values: 'values',
        parameters: 'parameters',
        layout: 'cell-major',
        cellCount: 3,
        sliceCount: 8,
        cost: 'mean',
        maximumChangePoints: 1,
        output: {changePointCount: 'count', changePoints: 'points', optimalCost: 'cost'}
      })
    ],
    outputs: ['count', 'points', 'cost']
  });
  expectNumericArrayClose(actual['count'], [0, 0, 0]);
  expectNumericArrayClose(actual['points'], [0xffffffff, 0xffffffff, 0xffffffff]);
  expectNumericArrayClose(actual['cost'], WIDE_MEAN_COST_EXPECTED, {relativeTolerance: 2e-6});
});

it('W5-21 preserves finite GPU mean costs when summing squared offsets would overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const actual = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: WIDE_MEAN_SQUARES_VALUES},
      parameters: {format: 'float32', values: Float32Array.of(1, 5)}
    },
    operations: [
      new GPULocationChangePointsOperation({
        values: 'values',
        parameters: 'parameters',
        layout: 'cell-major',
        cellCount: 4,
        sliceCount: 5,
        cost: 'mean',
        maximumChangePoints: 1,
        output: {changePointCount: 'count', changePoints: 'points', optimalCost: 'cost'}
      })
    ],
    outputs: ['count', 'points', 'cost']
  });
  expectNumericArrayClose(actual['count'], [0, 0, 0, 0]);
  expectNumericArrayClose(actual['points'], [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
  expectNumericArrayClose(actual['cost'], WIDE_MEAN_SQUARES_EXPECTED, {relativeTolerance: 2e-6});
});

for (const fixture of WIDE_STANDARDIZATION_CASES) {
  it(`B7-CP-02 GPU standardizes wide ${fixture.cost} inputs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const actual = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: fixture.values},
        parameters: {format: 'float32', values: Float32Array.of(1, fixture.minimumLength)}
      },
      operations: [
        new GPULocationChangePointsOperation({
          values: 'values',
          parameters: 'parameters',
          layout: 'cell-major',
          cellCount: fixture.counts.length,
          sliceCount: fixture.sliceCount,
          cost: fixture.cost,
          maximumChangePoints: 1,
          output: {
            changePointCount: 'count',
            requiredChangePointCounts: 'required',
            changePoints: 'points',
            optimalCost: 'cost',
            status: {overflow: 'overflow'}
          }
        })
      ],
      outputs: ['count', 'required', 'points', 'cost', 'overflow']
    });
    expectNumericArrayClose(actual['count'], fixture.counts);
    expectNumericArrayClose(actual['required'], fixture.counts);
    expectNumericArrayClose(actual['points'], fixture.points);
    expectNumericArrayClose(actual['overflow'], [0]);
    expectNumericArrayClose(actual['cost'], fixture.expectedCost, {relativeTolerance: 2e-5});
  });
}

for (const fixture of REVIEW_TWO_CHANGE_CASES) {
  it(`review 2 GPU ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const actual = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: fixture.values},
        parameters: {format: 'float32', values: fixture.parameters}
      },
      operations: [
        new GPULocationChangePointsOperation({
          values: 'values',
          parameters: 'parameters',
          layout: 'cell-major',
          cellCount: fixture.points.length,
          sliceCount: fixture.sliceCount,
          cost: fixture.cost,
          maximumChangePoints: 1,
          output: {
            changePointCount: 'count',
            requiredChangePointCounts: 'required',
            changePoints: 'points',
            optimalCost: 'cost',
            status: {overflow: 'overflow'}
          }
        })
      ],
      outputs: ['count', 'required', 'points', 'cost', 'overflow']
    });
    expectNumericArrayClose(actual['points'], fixture.points);
    expectNumericArrayClose(
      actual['count'],
      fixture.points.map(() => 1)
    );
    expectNumericArrayClose(actual['required'], fixture.required);
    expectNumericArrayClose(actual['cost'], fixture.expected, {
      relativeTolerance: fixture.relativeTolerance
    });
    expectNumericArrayClose(actual['overflow'], [fixture.overflow]);
  });
}

it('retains small mean changes and reports overflow after large mixed-sign observations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPULocationChangePointsOperation, CPULocationChangePointsOperation]) {
    const actual = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.of(1e7, -1e7, 1, 1, 1.25, 1.25, 1, 1)},
        parameters: {format: 'float32', values: Float32Array.of(1 / 64, 1)}
      },
      operations: [
        new Operation({
          values: 'values',
          parameters: 'parameters',
          layout: 'cell-major',
          cellCount: 1,
          sliceCount: 8,
          cost: 'mean',
          maximumChangePoints: 2,
          output: {
            changePointCount: 'count',
            changePoints: 'points',
            requiredChangePointCounts: 'required',
            optimalCost: 'cost',
            status: {overflow: 'overflow'}
          }
        })
      ],
      outputs: ['count', 'points', 'required', 'cost', 'overflow']
    });
    // Exhaustive Decimal partitions: four zero-RSS segments after the two outliers, 4 penalties.
    expectNumericArrayClose(actual['count'], [2]);
    expectNumericArrayClose(actual['points'], [1, 2]);
    expectNumericArrayClose(actual['required'], [4]);
    expectNumericArrayClose(actual['cost'], [0.0625], {absoluteTolerance: 1e-7});
    expectNumericArrayClose(actual['overflow'], [1]);
  }
});

it('does not invent variance changes when the absolute mean rounds onto an input group', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPULocationChangePointsOperation, CPULocationChangePointsOperation]) {
    for (const offset of [0, 1000000]) {
      const actual = await runOperationGraph(device, {
        inputs: {
          values: {
            format: 'float32',
            values: Float32Array.from(
              [0, 0, 0, 0, 0.0625, 0.0625, 0.0625, 0.0625],
              value => value + offset
            )
          },
          parameters: {format: 'float32', values: Float32Array.of(3, 2)}
        },
        operations: [
          new Operation({
            values: 'values',
            parameters: 'parameters',
            layout: 'cell-major',
            cellCount: 1,
            sliceCount: 8,
            cost: 'variance',
            maximumChangePoints: 2,
            output: {
              changePointCount: 'count',
              changePoints: 'points',
              requiredChangePointCounts: 'required',
              optimalCost: 'cost',
              status: {overflow: 'overflow'}
            }
          })
        ],
        outputs: ['count', 'points', 'required', 'cost', 'overflow']
      });
      // Every deviation from the true mean is +/-1/32, so every split only adds penalty.
      expectNumericArrayClose(actual['count'], [0]);
      expectNumericArrayClose(actual['points'], [0xffffffff, 0xffffffff]);
      expectNumericArrayClose(actual['required'], [0]);
      expectNumericArrayClose(actual['cost'], [-32.74875791352086], {absoluteTolerance: 2e-5});
      expectNumericArrayClose(actual['overflow'], [0]);
    }
  }
});

it('audits all five segment costs and every GPU/CPU operation output against exhaustive partition literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {cost, minimumLength, expected} of AUDIT_CHANGE_CASES) {
    for (const layout of ['cell-major', 'period-major'] as const) {
      for (const affine of [false, true]) {
        const penalty = affine
          ? cost === 'poisson'
            ? 6
            : cost === 'mean' || cost === 'slope'
              ? 12
              : 3
          : 3;
        const target = {
          ...expected,
          optimalCost: affine ? AUDIT_CHANGE_AFFINE_COSTS[cost] : expected.optimalCost
        };
        for (const Operation of [
          GPULocationChangePointsOperation,
          CPULocationChangePointsOperation
        ]) {
          const actual = await runOperationGraph(device, {
            inputs: {
              values: {
                format: 'float32',
                values: packAuditSeries(AUDIT_CHANGE_SERIES, layout, value =>
                  affine ? 2 * value + (cost === 'poisson' ? 0 : 3) : value
                )
              },
              parameters: {format: 'float32', values: Float32Array.of(penalty, minimumLength)}
            },
            operations: [
              new Operation({
                values: 'values',
                parameters: 'parameters',
                layout,
                cellCount: 5,
                sliceCount: 8,
                cost,
                maximumChangePoints: 1,
                output: {
                  changePointCount: 'changePointCount',
                  changePoints: 'changePoints',
                  requiredChangePointCounts: 'requiredChangePointCounts',
                  optimalCost: 'optimalCost',
                  status: {overflow: 'overflow'}
                }
              })
            ],
            outputs: Object.keys(expected)
          });
          for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
            // f32 standardized moments, logs and the deviance polynomial: 3e-4 relative cost.
            // Counts, original slice positions, capacity tails and overflow status are exact.
            expectNumericArrayClose(actual[port], target[port], {
              absoluteTolerance: port === 'optimalCost' ? 1e-4 : 0,
              relativeTolerance: port === 'optimalCost' ? 3e-4 : 0,
              label: `${cost}, ${layout}, ${affine}, ${port}`
            });
          }
        }
      }
    }
  }
});

it('audits all Pettitt GPU/CPU outputs using cross-split signs rather than twin equality', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const layout of ['cell-major', 'period-major'] as const) {
    for (const affine of [false, true]) {
      for (const Operation of [GPUPettittTestOperation, CPUPettittTestOperation]) {
        const actual = await runOperationGraph(device, {
          inputs: {
            values: {
              format: 'float32',
              values: packAuditSeries(AUDIT_PETTITT_SERIES, layout, value =>
                affine ? 2 * value + 3 : value
              )
            }
          },
          operations: [
            new Operation({
              values: 'values',
              layout,
              cellCount: 5,
              sliceCount: 8,
              output: {statistic: 'statistic', changeSlice: 'changeSlice', pValues: 'pValues'}
            })
          ],
          outputs: Object.keys(AUDIT_PETTITT_EXPECTED)
        });
        for (const port of Object.keys(
          AUDIT_PETTITT_EXPECTED
        ) as (keyof typeof AUDIT_PETTITT_EXPECTED)[]) {
          // Only exp and its f32 argument need tolerance; statistics and slice indices are exact.
          expectNumericArrayClose(actual[port], AUDIT_PETTITT_EXPECTED[port], {
            absoluteTolerance: port === 'pValues' ? 2e-6 : 0,
            label: `${layout}, ${port}`
          });
        }
      }
    }
  }
});

it('keeps an unsplit residual under a huge penalty and suppresses invalid Poisson input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const cost of ['mean', 'poisson'] as const) {
    const actual = await runOperationGraph(device, {
      inputs: {
        values: {
          format: 'float32',
          values: cost === 'mean' ? Float32Array.of(0, 1) : Float32Array.of(-1, 2)
        },
        parameters: {format: 'float32', values: Float32Array.of(2 ** 60, 1)}
      },
      operations: [
        new GPULocationChangePointsOperation({
          values: 'values',
          parameters: 'parameters',
          layout: 'cell-major',
          cellCount: 1,
          sliceCount: 2,
          cost,
          maximumChangePoints: 1,
          output: {
            changePointCount: 'changePointCount',
            changePoints: 'changePoints',
            requiredChangePointCounts: 'requiredChangePointCounts',
            optimalCost: 'optimalCost',
            status: {overflow: 'overflow'}
          }
        })
      ],
      outputs: [
        'changePointCount',
        'changePoints',
        'requiredChangePointCounts',
        'optimalCost',
        'overflow'
      ]
    });
    expectNumericArrayClose(actual['optimalCost'], cost === 'mean' ? [0.5] : [NaN]);
    expectNumericArrayClose(actual['changePointCount'], [0]);
    expectNumericArrayClose(actual['changePoints'], [0xffffffff]);
    expectNumericArrayClose(actual['requiredChangePointCounts'], [0]);
    expectNumericArrayClose(actual['overflow'], [0]);
  }
});
