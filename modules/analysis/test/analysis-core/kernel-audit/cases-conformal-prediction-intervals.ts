// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {KernelAuditCase} from './kernel-audit-context';
import {
  GPUConformalPredictionIntervals,
  GPUGeoConformalPredictionIntervals
} from '../../../src/gpu-spatial-analysis/spatial-validation/index';

export const CONFORMAL_PREDICTION_INTERVALS_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUConformalPredictionIntervals',
    name: 'split-maximum-rows',
    build({graph, view}) {
      graph.add(
        new GPUConformalPredictionIntervals({
          calibration: {
            observations: view('float32', 1 << 20),
            predictions: view('float32', 1 << 20),
            scales: view('float32', 1 << 20),
            mask: view('uint32', 1 << 20)
          },
          query: {predictions: view('float32', 1 << 20), scales: view('float32', 1 << 20)},
          parameters: view('float32', 4),
          output: {
            intervals: view('float32', 2 << 20),
            status: view('uint32', 1 << 20),
            calibrationCount: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUConformalPredictionIntervals',
    name: 'cv-plus-maximum-folds',
    build({graph, view}) {
      graph.add(
        new GPUConformalPredictionIntervals({
          calibration: {
            observations: view('float32', 65536),
            predictions: view('float32', 64 * 65536),
            folds: view('uint32', 65536),
            scales: view('float32', 65536),
            mask: view('uint32', 65536)
          },
          query: {predictions: view('float32', 64 * 65536), scales: view('float32', 65536)},
          foldCount: 64,
          parameters: view('float32', 4),
          output: {
            intervals: view('float32', 2 * 65536),
            status: view('uint32', 65536),
            calibrationCount: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGeoConformalPredictionIntervals',
    name: 'maximum-neighbours',
    build({graph, view}) {
      graph.add(
        new GPUGeoConformalPredictionIntervals({
          calibration: {
            observations: view('float32', 65536),
            predictions: view('float32', 65536),
            scales: view('float32', 65536),
            mask: view('uint32', 65536)
          },
          query: {predictions: view('float32', 65536), scales: view('float32', 65536)},
          neighbors: {
            offsets: view('uint32', 65537),
            indices: view('uint32', 65536 * 256),
            weights: view('float32', 65536 * 256)
          },
          parameters: view('float32', 4),
          output: {intervals: view('float32', 2 * 65536), status: view('uint32', 65536)}
        })
      );
    }
  }
];
