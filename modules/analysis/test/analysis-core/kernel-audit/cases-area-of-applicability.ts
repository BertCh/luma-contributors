// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {KernelAuditCase} from './kernel-audit-context';
import {GPUAreaOfApplicability} from '../../../src/gpu-spatial-analysis/spatial-validation/index';

export const AREA_OF_APPLICABILITY_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUAreaOfApplicability',
    name: 'maximum-rows-predictors-folds',
    build({graph, view}) {
      const rows = 65536;
      const queryRows = 65536;
      graph.add(
        new GPUAreaOfApplicability({
          training: view('float32', rows * 16),
          query: view('float32', queryRows * 16),
          predictorCount: 16,
          weights: view('float32', 16),
          mask: view('uint32', rows),
          crossValidation: {
            folds: view('uint32', rows),
            training: view('uint32', 64 * rows),
            foldCount: 64
          },
          output: {
            trainDissimilarity: view('float32', rows),
            queryDissimilarity: view('float32', queryRows),
            applicable: view('uint32', queryRows),
            summary: view('float32', 6),
            status: view('uint32', 1)
          }
        })
      );
    }
  }
];
