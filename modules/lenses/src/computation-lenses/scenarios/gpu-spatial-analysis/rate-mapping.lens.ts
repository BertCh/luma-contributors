// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

for (const [element, slug, title] of [
  [
    'addExcessRiskRecipe',
    'excess-risk-recipe',
    'Observed counts relative to pooled expected counts'
  ],
  ['addSpatialRateRecipe', 'spatial-rate-recipe', 'Population pooled neighborhood rates']
] as const) {
  const id = `gpu-spatial-analysis.${slug}`;
  defineLensScenario({
    id,
    entry: 'gpu-spatial-analysis',
    element,
    title,
    audience: 'engineer',
    rungs: ['result', 'datum', 'kernel', 'graph'],
    defaultRung: 'graph',
    dataset: {
      id: 'rate-mapping-illustrative/1',
      source:
        'Repository-authored three-row counts and populations; a path neighborhood with unequal exposures.',
      licence: 'MIT (repository-authored illustrative inputs)'
    },
    async load() {
      const [{runS2Lens}, {buildRateMappingScenario}] = await Promise.all([
        import('./s2-lens-run'),
        import('./rate-mapping-examples')
      ]);
      return {
        run: device =>
          runS2Lens(device, {
            scenarioId: id,
            element,
            datasets: ['rate-mapping-illustrative/1'],
            build: (graph, uploads) => buildRateMappingScenario(element, graph, uploads)
          })
      };
    }
  });
}
