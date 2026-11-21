// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenarios} from '../../define-lens-scenarios';
import {getPointProcessKeySpaces} from './point-process-domains';
import type {GPUComputationAttributeValue} from '@luma-contributors/analysis/computation-record';

/** Choice receipts for the executed point-process branch. */
export function getPointProcessLensChoices(
  element: string,
  choices: Readonly<Record<string, GPUComputationAttributeValue>>,
  suppliedIntensity = false
): Readonly<Record<string, GPUComputationAttributeValue>> {
  if (element === 'GPUInhomogeneousPointPattern' && suppliedIntensity) {
    return {
      ...choices,
      bandwidth: 'disabled: supplied intensity',
      selfContribution: 'disabled: supplied intensity'
    };
  }
  if (element === 'GPUOPTICS') {
    const method = choices['extractionMethod'] ?? 'dbscan';
    return {
      ...choices,
      ...(method === 'xi'
        ? {extractionDistance: 'disabled: xi extraction'}
        : {xi: 'disabled: dbscan extraction', minimumClusterSize: 'disabled: dbscan extraction'})
    };
  }
  if (element === 'GPUPointPatternEnvelope') {
    return {
      ...choices,
      ...(choices['statistic'] === 'extreme-rank'
        ? {rank: 'disabled: extreme-rank envelope'}
        : {significanceLevel: `disabled: ${choices['statistic'] ?? 'maximum-deviation'}`})
    };
  }
  return choices;
}

registerLensScenarios(
  {
    entry: 'gpu-spatial-analysis',
    audience: 'analyst',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    defaultRung: 'result',
    dataset: {
      id: 'point-process-illustrative/1',
      source:
        'Repository-authored small planar or network frame; supplied intensity and supplied null curves are explicit illustrative inputs.',
      licence: 'MIT'
    }
  },
  [
    {
      id: 'gpu-spatial-analysis.inhomogeneous-point-pattern',
      element: 'GPUInhomogeneousPointPattern',
      title: 'GPUInhomogeneousPointPattern: explicit small point-process example',
      async load() {
        const [{runS2Lens}, {buildPointProcessScenario}] = await Promise.all([
          import('./s2-lens-run'),
          import('./point-process-examples')
        ]);
        return {
          run: device =>
            runS2Lens(device, {
              scenarioId: 'gpu-spatial-analysis.inhomogeneous-point-pattern',
              element: 'GPUInhomogeneousPointPattern',
              choices: getPointProcessLensChoices(
                'GPUInhomogeneousPointPattern',
                {
                  bandwidth: 'disabled: supplied intensity',
                  edgeCorrection: 'border',
                  pairCorrelation: 'annular finite difference',
                  selfContribution: 'exclude'
                },
                true
              ),
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: getPointProcessKeySpaces('GPUInhomogeneousPointPattern'),
              build: (graph, uploads) =>
                buildPointProcessScenario('GPUInhomogeneousPointPattern', graph, uploads)
            })
        };
      }
    },
    {
      id: 'gpu-spatial-analysis.point-pattern-envelope',
      element: 'GPUPointPatternEnvelope',
      title: 'GPUPointPatternEnvelope: explicit small point-process example',
      async load() {
        const [{runS2Lens}, {buildPointProcessScenario}] = await Promise.all([
          import('./s2-lens-run'),
          import('./point-process-examples')
        ]);
        return {
          run: device =>
            runS2Lens(device, {
              scenarioId: 'gpu-spatial-analysis.point-pattern-envelope',
              element: 'GPUPointPatternEnvelope',
              choices: getPointProcessLensChoices('GPUPointPatternEnvelope', {
                statistic: 'integrated-squared-deviation',
                rank: 1,
                significanceLevel: 0.05,
                simulationCount: 3,
                fittedParameters: 'none'
              }),
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: getPointProcessKeySpaces('GPUPointPatternEnvelope'),
              build: (graph, uploads) =>
                buildPointProcessScenario('GPUPointPatternEnvelope', graph, uploads)
            })
        };
      }
    },
    {
      id: 'gpu-spatial-analysis.spatial-balanced-sample',
      element: 'GPUSpatialBalancedSample',
      title: 'GPUSpatialBalancedSample: explicit small point-process example',
      async load() {
        const [{runS2Lens}, {buildPointProcessScenario}] = await Promise.all([
          import('./s2-lens-run'),
          import('./point-process-examples')
        ]);
        return {
          run: device =>
            runS2Lens(device, {
              scenarioId: 'gpu-spatial-analysis.spatial-balanced-sample',
              element: 'GPUSpatialBalancedSample',
              choices: {seed: 7, replicate: 0, allocations: 'stratum 0: 2'},
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: getPointProcessKeySpaces('GPUSpatialBalancedSample'),
              build: (graph, uploads) =>
                buildPointProcessScenario('GPUSpatialBalancedSample', graph, uploads)
            })
        };
      }
    },
    {
      id: 'gpu-spatial-analysis.optics',
      element: 'GPUOPTICS',
      title: 'GPUOPTICS: explicit small point-process example',
      async load() {
        const [{runS2Lens}, {buildPointProcessScenario}] = await Promise.all([
          import('./s2-lens-run'),
          import('./point-process-examples')
        ]);
        return {
          run: device =>
            runS2Lens(device, {
              scenarioId: 'gpu-spatial-analysis.optics',
              element: 'GPUOPTICS',
              choices: getPointProcessLensChoices('GPUOPTICS', {
                minimumSamples: 2,
                maximumDistance: 2,
                extractionDistance: 1.5,
                extractionMethod: 'dbscan',
                xi: 0.05,
                minimumClusterSize: 2,
                neighborIndex: 'auto',
                orderingMethod: 'sequential',
                frontierBatchSize: 64,
                maximumIterations: 4
              }),
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: getPointProcessKeySpaces('GPUOPTICS'),
              build: (graph, uploads) => buildPointProcessScenario('GPUOPTICS', graph, uploads)
            })
        };
      }
    }
  ]
);
