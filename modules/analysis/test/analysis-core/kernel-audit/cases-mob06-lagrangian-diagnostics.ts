// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFiniteTimeLyapunovExponent,
  GPULagrangianFlowMap
} from '../../../src/raster-analysis/lagrangian-diagnostics/index';
import type {KernelAuditCase} from './kernel-audit-context';

const FIELD_SIDE = 2048;

/** Kernel-audit cases of the Lagrangian flow map and the FTLE (MOB-06). */
export const MOB06_LAGRANGIAN_DIAGNOSTICS_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPULagrangianFlowMap',
    name: 'regions, connectivity classes and residence at the maximum step count',
    build: ({graph, view, maximumRows}) => {
      const seedCount = maximumRows(8);
      graph.add(
        new GPULagrangianFlowMap({
          seeds: view('float32x2', seedCount),
          velocitiesStart: view('float32x2', FIELD_SIDE * FIELD_SIDE),
          velocitiesEnd: view('float32x2', FIELD_SIDE * FIELD_SIDE),
          fieldWidth: FIELD_SIDE,
          fieldHeight: FIELD_SIDE,
          stepCount: 8192,
          parameters: view('float32', 8),
          regions: view('uint32', FIELD_SIDE * FIELD_SIDE),
          output: {
            positions: view('float32x2', seedCount),
            regionClasses: view('uint32', 2 * seedCount),
            residenceTimes: view('float32', seedCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULagrangianFlowMap',
    name: 'positions only',
    build: ({graph, view}) =>
      graph.add(
        new GPULagrangianFlowMap({
          seeds: view('float32x2', 4096),
          velocitiesStart: view('float32x2', 64 * 64),
          velocitiesEnd: view('float32x2', 64 * 64),
          fieldWidth: 64,
          fieldHeight: 64,
          stepCount: 100,
          parameters: view('float32', 8),
          output: {positions: view('float32x2', 4096)}
        })
      )
  },
  {
    exportName: 'GPUFiniteTimeLyapunovExponent',
    name: 'largest seed grid',
    build: ({graph, view}) =>
      graph.add(
        new GPUFiniteTimeLyapunovExponent({
          positions: view('float32x2', FIELD_SIDE * FIELD_SIDE),
          seedWidth: FIELD_SIDE,
          seedHeight: FIELD_SIDE,
          parameters: view('float32', 4),
          output: {ftle: view('float32', FIELD_SIDE * FIELD_SIDE)}
        })
      )
  }
];
