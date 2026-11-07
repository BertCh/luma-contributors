// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {GPUNestedInterpolationValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import type {KernelAuditCase} from './kernel-audit-context';
export const SPATIAL_VALIDATION_ADVANCED_KERNEL_CASES: KernelAuditCase[] = [
  ...(['groups', 'environmental', 'nndm', 'knndm'] as const).map(
    mode =>
      ({
        exportName: 'GPUSpatialCrossValidation',
        name: `advanced-${mode}-capacity`,
        build({graph, view}) {
          const rows = mode === 'nndm' ? 64 : 256,
            foldCount = mode === 'nndm' ? 64 : 16;
          graph.add(
            new GPUSpatialCrossValidation({
              positions: view('float32x2', rows),
              blocks: view('uint32', rows),
              mask: view('uint32', rows),
              foldCount,
              design:
                mode === 'groups'
                  ? {mode}
                  : mode === 'environmental'
                    ? {mode, featureCount: 8, features: view('float32', rows * 8)}
                    : mode === 'nndm'
                      ? {mode, predictionPositions: view('float32x2', 256), maximumDistance: 8}
                      : {
                          mode,
                          predictionPositions: view('float32x2', 256),
                          candidates: view('uint32', rows * 64),
                          candidateCount: 64
                        },
              output: {
                folds: view('uint32', rows),
                training: view('uint32', rows * foldCount),
                testing: view('uint32', rows * foldCount),
                counts: view('uint32', foldCount * 3),
                status: view('uint32', foldCount)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  {
    exportName: 'GPUNestedInterpolationValidation',
    name: 'both-models-maximal-local-system',
    build({graph, view}) {
      graph.add(
        new GPUNestedInterpolationValidation({
          positions: view('float32x2', 64),
          response: view('float32', 64),
          training: view('uint32', 128),
          testing: view('uint32', 128),
          foldCount: 2,
          innerBlocks: view('uint32', 64),
          innerFoldCount: 2,
          neighborCount: 32,
          candidates: [
            {mode: 'ordinary-kriging', covariance: {model: 'gaussian', range: 3, sill: 1}},
            {mode: 'rbf', kernel: 'quintic', degree: 2, epsilon: 1}
          ],
          output: {
            selected: view('uint32', 2),
            scores: view('float32', 12),
            predictions: view('float32', 128),
            metrics: view('float32', 12),
            status: view('uint32', 2)
          }
        })
      );
    }
  }
];
