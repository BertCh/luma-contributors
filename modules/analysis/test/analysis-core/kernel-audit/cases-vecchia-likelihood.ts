// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUVecchiaFisherScoring,
  GPUVecchiaLogLikelihood,
  GPUVecchiaSpaceTimeLogLikelihood,
  GPU_VECCHIA_FISHER_CELL_COUNT,
  GPU_VECCHIA_PARAMETER_LENGTH,
  GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/index';
import type {KernelAuditCase} from './kernel-audit-context';

const ROW_COUNT = 1024;
const NEIGHBOR_COUNT = 32;

export const VECCHIA_LIKELIHOOD_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUVecchiaLogLikelihood',
    name: 'maximum-conditioning-set',
    build({graph, view}) {
      graph.add(
        new GPUVecchiaLogLikelihood({
          positions: view('float32x2', ROW_COUNT),
          observations: view('float32x4', ROW_COUNT),
          neighborhoodOffsets: view('uint32', ROW_COUNT + 1),
          neighborhoods: view('uint32', ROW_COUNT * NEIGHBOR_COUNT),
          neighborCount: NEIGHBOR_COUNT,
          parameters: view('float32', GPU_VECCHIA_PARAMETER_LENGTH),
          output: {
            conditionals: view('float32x4', ROW_COUNT),
            conditionalStatus: view('uint32', ROW_COUNT),
            siteTerms: view('float32', ROW_COUNT),
            siteStatus: view('uint32', ROW_COUNT),
            logLikelihood: view('float32', 2),
            status: view('uint32', 3)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUVecchiaFisherScoring',
    name: 'maximum-conditioning-set',
    build({graph, view}) {
      graph.add(
        new GPUVecchiaFisherScoring({
          positions: view('float32x2', ROW_COUNT),
          observations: view('float32x4', ROW_COUNT),
          neighborhoodOffsets: view('uint32', ROW_COUNT + 1),
          neighborhoods: view('uint32', ROW_COUNT * NEIGHBOR_COUNT),
          neighborCount: NEIGHBOR_COUNT,
          parameters: view('float32', GPU_VECCHIA_PARAMETER_LENGTH),
          output: {
            siteTerms: view('float32', ROW_COUNT * GPU_VECCHIA_FISHER_CELL_COUNT),
            siteStatus: view('uint32', ROW_COUNT),
            totals: view('float32', GPU_VECCHIA_FISHER_CELL_COUNT * 2),
            status: view('uint32', 3)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUVecchiaSpaceTimeLogLikelihood',
    name: 'maximum-conditioning-set',
    build({graph, view}) {
      graph.add(
        new GPUVecchiaSpaceTimeLogLikelihood({
          temporalContext: {
            unit: 'second',
            epoch: 0n,
            encoding: 'float32-relative',
            axis: {kind: 'irregular'}
          },
          positions: view('float32x2', ROW_COUNT),
          times: view('float32', ROW_COUNT),
          observations: view('float32x4', ROW_COUNT),
          neighborhoodOffsets: view('uint32', ROW_COUNT + 1),
          neighborhoods: view('uint32', ROW_COUNT * NEIGHBOR_COUNT),
          neighborCount: NEIGHBOR_COUNT,
          parameters: view('float32', GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH),
          output: {
            conditionals: view('float32x4', ROW_COUNT),
            conditionalStatus: view('uint32', ROW_COUNT),
            siteTerms: view('float32', ROW_COUNT),
            siteStatus: view('uint32', ROW_COUNT),
            logLikelihood: view('float32', 2),
            status: view('uint32', 3)
          }
        })
      );
    }
  }
];
