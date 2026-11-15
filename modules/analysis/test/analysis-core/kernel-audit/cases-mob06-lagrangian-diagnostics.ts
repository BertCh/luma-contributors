// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUFiniteTimeLyapunovExponent,
  GPUScalarFieldRidges,
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
      const seedCount = maximumRows(16);
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
            validity: view('uint32', seedCount),
            regionClasses: view('uint32', 2 * seedCount),
            residenceTimes: view('float32', seedCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPULagrangianFlowMap',
    name: 'geographic positions with all diagnostics',
    build: ({graph, view}) =>
      graph.add(
        new GPULagrangianFlowMap({
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
          seeds: view('float32x2', 4096),
          velocitiesStart: view('float32x2', 64 * 64),
          velocitiesEnd: view('float32x2', 64 * 64),
          fieldWidth: 64,
          fieldHeight: 64,
          stepCount: 100,
          parameters: view('float32', 8),
          regions: view('uint32', 64 * 64),
          output: {
            positions: view('float32x2', 4096),
            validity: view('uint32', 4096),
            regionClasses: view('uint32', 8192),
            residenceTimes: view('float32', 4096)
          }
        })
      )
  },
  {
    exportName: 'GPUFiniteTimeLyapunovExponent',
    name: 'largest geographic seed grid with explicit validity',
    build: ({graph, view}) =>
      graph.add(
        new GPUFiniteTimeLyapunovExponent({
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
          validity: view('uint32', FIELD_SIDE * FIELD_SIDE),
          positions: view('float32x2', FIELD_SIDE * FIELD_SIDE),
          seedWidth: FIELD_SIDE,
          seedHeight: FIELD_SIDE,
          parameters: view('float32', 4),
          output: {
            ftle: view('float32', FIELD_SIDE * FIELD_SIDE),
            validity: view('uint32', FIELD_SIDE * FIELD_SIDE)
          }
        })
      )
  },
  {
    exportName: 'GPUScalarFieldRidges',
    name: 'largest binding-sized geographic field, missing slots and strength',
    build: ({graph, view, maximumRows}) => {
      const width = 8192;
      const height = Math.floor(maximumRows(4) / width);
      const count = width * height;
      graph.add(
        new GPUScalarFieldRidges({
          input: {
            id: 'density',
            format: 'float32',
            storage: {kind: 'buffer', values: view('float32', count)},
            validity: view('uint32', count)
          },
          grid: {width, height, affine: [0.01, 0, 0, 0, -0.01, 45], pixelInterpretation: 'area'},
          cellSizeMode: 'geographic',
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
          parameters: view('float32', 8),
          output: {
            ridges: view('uint32', count),
            validity: view('uint32', count),
            strength: view('float32', count)
          }
        })
      );
    }
  }
];
