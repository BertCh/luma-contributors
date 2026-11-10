// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import type {KernelAuditCase} from './kernel-audit-context';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression
} from '../../../src/gpu-spatial-analysis/spatial-regression/index';

const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

export const GTWR_AUDIT_CASES: KernelAuditCase[] = [
  ...(['gaussian', 'bisquare'] as const).map(kernel => ({
    exportName: 'GPUGeographicallyTemporallyWeightedRegression',
    name: `maximum-rows-predictors-${kernel}`,
    build({graph, view}: Parameters<KernelAuditCase['build']>[0]) {
      const rows = 32768;
      graph.add(
        new GPUGeographicallyTemporallyWeightedRegression({
          positions: view('float32x2', rows),
          times: view('float32', rows),
          temporalContext: TEMPORAL_CONTEXT,
          predictors: view('float32', rows * 11),
          predictorCount: 11,
          response: view('float32', rows),
          mask: view('uint32', rows),
          kernel,
          parameters: view('float32', 4),
          output: {
            coefficients: view('float32', rows * 12),
            localStatus: view('uint32', rows),
            fitted: view('float32', rows),
            hatDiagonal: view('float32', rows),
            residuals: view('float32', rows),
            summary: view('float32', 12)
          }
        })
      );
    }
  })),
  {
    exportName: 'GPUMultiscaleGeographicallyTemporallyWeightedRegression',
    name: 'maximum-iterations-predictors',
    build({graph, view}) {
      const rows = 32768;
      graph.add(
        new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
          positions: view('float32x2', rows),
          times: view('float32', rows),
          temporalContext: TEMPORAL_CONTEXT,
          predictors: view('float32', rows * 11),
          predictorCount: 11,
          response: view('float32', rows),
          mask: view('uint32', rows),
          kernel: 'bisquare',
          iterationCount: 32,
          parameters: view('float32', 2 + 2 * 12),
          output: {
            coefficients: view('float32', rows * 12),
            fitted: view('float32', rows),
            residuals: view('float32', rows),
            localStatus: view('uint32', rows),
            scores: view('float32', 32)
          }
        })
      );
    }
  }
];
