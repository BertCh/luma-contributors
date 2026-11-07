// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

export const COVERAGE_TRANCHE_ONE_SCENARIOS = [
  {
    element: 'GPURasterFractionalZonalDistribution',
    slug: 'fractional-zonal-distribution',
    domain: 'gpu-raster',
    title: 'Fractional weighted quartiles and effective support'
  },
  {
    element: 'GPURasterFractionalCategoricalStatistics',
    slug: 'fractional-categorical-statistics',
    domain: 'gpu-raster',
    title: 'Land-cover fractions with explicit missing area'
  },
  {
    element: 'GPURasterFocalPairs',
    slug: 'focal-pairs',
    domain: 'gpu-raster',
    title: 'Local paired regression on an exact linear response'
  },
  {
    element: 'GPURasterNDWI',
    slug: 'ndwi',
    domain: 'gpu-raster',
    title: 'McFeeters open-water index on calibrated reflectance'
  },
  {
    element: 'GPURasterNBR',
    slug: 'nbr',
    domain: 'gpu-raster',
    title: 'Normalized burn ratio with an explicit SWIR2 band'
  },
  {
    element: 'GPURasterEVI',
    slug: 'evi',
    domain: 'gpu-raster',
    title: 'Enhanced vegetation index from three calibrated bands'
  },
  {
    element: 'GPURasterSAVI',
    slug: 'savi',
    domain: 'gpu-raster',
    title: 'Soil-adjusted vegetation index with L = 0.5'
  },
  {
    element: 'addDasymetricRedistributionRecipe',
    slug: 'dasymetric',
    domain: 'gpu-spatial-analysis',
    title: 'Population redistribution with a source mass ledger'
  }
] as const;

for (const {element, slug, domain, title} of COVERAGE_TRANCHE_ONE_SCENARIOS) {
  const elementId = domain === 'gpu-raster' ? `raster-${slug}` : 'dasymetric-redistribution-recipe';
  const id = `${domain}.${elementId}`;
  defineLensScenario({
    id,
    entry: domain,
    element,
    title,
    audience: 'analyst',
    rungs: ['result', 'graph', 'table', 'kernel'],
    defaultRung: 'result',
    example: {path: `scenarios/gpu-raster/coverage-tranche-one-${slug}.example.ts`},
    dataset: {
      id: 'coverage-tranche-one-synthetic/1',
      licence: 'MIT (repository-authored synthetic inputs)',
      source:
        'Explicit 3 × 3 planar grid with 100 m² cells; prepared fractional source coverage and scaled integer reflectance.'
    },
    load: async () => ({
      run: device =>
        import('./coverage-tranche-one-runner').then(module =>
          module.runCoverageTrancheOne(device, element, id)
        )
    })
  });
}
