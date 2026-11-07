// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenarios} from '../../define-lens-scenarios';

registerLensScenarios(
  {
    entry: 'gpu-spatial-analysis',
    audience: 'analyst',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    defaultRung: 'result',
    dataset: {
      id: 'synthetic-interpolation-six-sites-v1',
      source:
        'Repository-authored six planar sites with an affine response; simulation uses seed 347.',
      licence: 'MIT'
    }
  },
  [
    {
      id: 'gpu-spatial-analysis.local-kriging',
      element: 'GPULocalKriging',
      title: 'GPULocalKriging: synthetic planar interpolation',
      async load() {
        const {runInterpolationLens} = await import('./interpolation.run');
        return {run: device => runInterpolationLens(device, 'local-kriging')};
      }
    },
    {
      id: 'gpu-spatial-analysis.local-rbf',
      element: 'GPULocalRBF',
      title: 'GPULocalRBF: synthetic planar interpolation',
      async load() {
        const {runInterpolationLens} = await import('./interpolation.run');
        return {run: device => runInterpolationLens(device, 'local-rbf')};
      }
    },
    {
      id: 'gpu-spatial-analysis.natural-neighbor',
      element: 'GPUNaturalNeighbor',
      title: 'GPUNaturalNeighbor: synthetic planar interpolation',
      async load() {
        const {runInterpolationLens} = await import('./interpolation.run');
        return {run: device => runInterpolationLens(device, 'natural-neighbor')};
      }
    },
    {
      id: 'gpu-spatial-analysis.conditional-gaussian-simulation',
      element: 'GPUConditionalGaussianSimulation',
      title: 'GPUConditionalGaussianSimulation: synthetic planar interpolation',
      async load() {
        const {runInterpolationLens} = await import('./interpolation.run');
        return {run: device => runInterpolationLens(device, 'conditional-gaussian-simulation')};
      }
    }
  ]
);
