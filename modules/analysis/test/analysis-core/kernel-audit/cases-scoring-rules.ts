// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUEnsembleScores,
  GPUGaussianPredictiveScores
} from '../../../src/gpu-spatial-analysis/scoring-rules/index';
import type {KernelAuditCase} from './kernel-audit-context';

/** Kernel-audit cases of the proper scoring rules, at binding-limited maxima. */
export const SCORING_RULES_AUDIT_CASES: readonly KernelAuditCase[] = [
  ...[true, false].flatMap((fair): KernelAuditCase[] => [
    {
      exportName: 'GPUEnsembleScores',
      name: `fair=${fair} 4096 members, binding-limited sites, every score`,
      build: ({graph, view, maximumRows}) => {
        const dimension = Math.floor(maximumRows(4) / 4096);
        graph.add(
          new GPUEnsembleScores({
            estimator: fair ? 'fair' : 'empirical',
            memberCount: 4096,
            dimension,
            variogramOrder: 2,
            ensemble: view('float32', 4096 * dimension),
            observations: view('float32', dimension),
            variogramPairs: view('uint32x2', maximumRows(8)),
            variogramWeights: view('float32', maximumRows(8)),
            output: {
              crps: view('float32', dimension),
              energyScore: view('float32', 1),
              variogramScore: view('float32', 1)
            }
          })
        );
      }
    },
    {
      exportName: 'GPUEnsembleScores',
      name: `fair=${fair} two members, the most sites that fit one binding`,
      build: ({graph, view, maximumRows}) => {
        const dimension = Math.floor(maximumRows(4) / 2);
        graph.add(
          new GPUEnsembleScores({
            estimator: fair ? 'fair' : 'empirical',
            memberCount: 2,
            dimension,
            ensemble: view('float32', 2 * dimension),
            observations: view('float32', dimension),
            output: {crps: view('float32', dimension)}
          })
        );
      }
    }
  ]),
  {
    exportName: 'GPUGaussianPredictiveScores',
    name: 'binding-limited sites',
    build: ({graph, view, maximumRows}) => {
      const sites = maximumRows(4);
      graph.add(
        new GPUGaussianPredictiveScores({
          means: view('float32', sites),
          variances: view('float32', sites),
          observations: view('float32', sites),
          output: {crps: view('float32', sites)}
        })
      );
    }
  }
];
