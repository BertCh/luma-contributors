// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenarios} from '../../define-lens-scenarios';

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
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: [{id: 'entry', label: 'Point-process input/output row', indexSpace: '1d'}],
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
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: [{id: 'entry', label: 'Point-process input/output row', indexSpace: '1d'}],
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
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: [{id: 'entry', label: 'Point-process input/output row', indexSpace: '1d'}],
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
              datasets: ['point-process-illustrative/1'],
              seed: 7,
              keySpaces: [{id: 'entry', label: 'Point-process input/output row', indexSpace: '1d'}],
              build: (graph, uploads) => buildPointProcessScenario('GPUOPTICS', graph, uploads)
            })
        };
      }
    }
  ]
);
