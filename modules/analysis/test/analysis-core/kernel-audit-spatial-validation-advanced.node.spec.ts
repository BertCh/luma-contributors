// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {SPATIAL_VALIDATION_ADVANCED_KERNEL_CASES} from './kernel-audit/cases-spatial-validation-advanced';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';
registerKernelAudit('spatial-validation-advanced', SPATIAL_VALIDATION_ADVANCED_KERNEL_CASES);

import {expect, it} from 'vitest';
import {createKernelAuditContext} from './kernel-audit/kernel-audit-context';
import {GPUSpatialCrossValidation} from '../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GPUSpatialCrossValidatedOLS} from '../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';
it('rejects OLS fit status aliasing later metric status', () => {
  const context = createKernelAuditContext('validation-ols-output-alias');
  const fitStatus = context.view('uint32', 2);
  try {
    expect(
      () =>
        new GPUSpatialCrossValidatedOLS({
          folds: {
            positions: context.view('float32x2', 8),
            blocks: context.view('uint32', 8),
            foldCount: 2,
            output: {
              folds: context.view('uint32', 8),
              training: context.view('uint32', 16),
              testing: context.view('uint32', 16),
              counts: context.view('uint32', 6),
              status: context.view('uint32', 2)
            }
          },
          predictors: context.view('float32', 8),
          predictorCount: 1,
          response: context.view('float32', 8),
          output: {
            coefficients: context.view('float32', 4),
            predictions: context.view('float32', 16),
            metrics: context.view('float32', 12),
            fitStatus,
            metricStatus: fitStatus
          }
        })
    ).toThrow(
      /^spatial-cross-validated-ols fitStatus and metricStatus logical writable byte ranges must not overlap$/
    );
  } finally {
    context.device.destroy();
  }
});
for (const design of [undefined, {mode: 'groups' as const}]) {
  it(`rejects overlapping training and testing masks with ${design ? 'group' : 'hashed'} assignment`, () => {
    const context = createKernelAuditContext('validation-mask-alias');
    const training = context.view('uint32', 16);
    try {
      expect(
        () =>
          new GPUSpatialCrossValidation({
            positions: context.view('float32x2', 8),
            blocks: context.view('uint32', 8),
            foldCount: 2,
            design,
            output: {
              folds: context.view('uint32', 8),
              training,
              testing: training,
              counts: context.view('uint32', 6),
              status: context.view('uint32', 2)
            }
          })
      ).toThrow(
        /^spatial-cross-validation training and testing logical writable byte ranges must not overlap$/
      );
    } finally {
      context.device.destroy();
    }
  });
}
it('rejects diagnostics aliasing a later counts output', () => {
  const context = createKernelAuditContext('design-output-alias');
  const counts = context.view('uint32', 6);
  const diagnostics = context.graph.createDataView(counts.buffer, {format: 'float32', length: 4});
  expect(
    () =>
      new GPUSpatialCrossValidation({
        positions: context.view('float32x2', 8),
        blocks: context.view('uint32', 8),
        foldCount: 2,
        design: {mode: 'groups', diagnostics},
        output: {
          folds: context.view('uint32', 8),
          training: context.view('uint32', 16),
          testing: context.view('uint32', 16),
          counts,
          status: context.view('uint32', 2)
        }
      })
  ).toThrow(
    /^spatial-cross-validation counts and diagnostics logical writable byte ranges must not overlap$/
  );
  context.device.destroy();
});

for (const alias of ['design-input', 'design-diagnostic'] as const) {
  it(`rejects OLS result aliasing a ${alias} buffer`, () => {
    const context = createKernelAuditContext(`validation-ols-${alias}`);
    const coefficients = context.view('float32', 4);
    const predictions = context.view('float32', 16);
    try {
      expect(
        () =>
          new GPUSpatialCrossValidatedOLS({
            folds: {
              positions: context.view('float32x2', 8),
              blocks: context.view('uint32', 8),
              foldCount: 2,
              design:
                alias === 'design-input'
                  ? {mode: 'environmental', features: predictions, featureCount: 2}
                  : {mode: 'groups', diagnostics: coefficients},
              output: {
                folds: context.view('uint32', 8),
                training: context.view('uint32', 16),
                testing: context.view('uint32', 16),
                counts: context.view('uint32', 6),
                status: context.view('uint32', 2)
              }
            },
            predictors: context.view('float32', 8),
            predictorCount: 1,
            response: context.view('float32', 8),
            output: {
              coefficients,
              predictions,
              metrics: context.view('float32', 12),
              fitStatus: context.view('uint32', 2),
              metricStatus: context.view('uint32', 2)
            }
          })
      ).toThrow(
        alias === 'design-input'
          ? 'share buffers with inputs'
          : /^spatial-cross-validated-ols coefficients and diagnostics logical writable byte ranges must not overlap$/
      );
    } finally {
      context.device.destroy();
    }
  });
}
