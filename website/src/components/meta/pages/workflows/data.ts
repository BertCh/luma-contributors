// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Code on the workflows page. The lens scenario files are the exact code behind each recorded run
// (their `// #region example` block); the drainage example is a page file the website typecheck
// compiles; the recipe result is cut from the real type declarations at build time.

import recipeUtilitiesSource from '../../../../../../modules/analysis/src/gpu-spatial-analysis/recipes/recipe-utils?raw';
import statusSource from '../../../../../../modules/analysis/src/analysis-core/contracts/status?raw';
import driveTimeExample from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-network/drive-time-catchment-recipe.example?raw';
import hotSpotsExample from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/hot-spot-analysis-recipe.example?raw';
import localMoranExample from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/local-moran.example?raw';
import rateClustersExample from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/rate-cluster-map-recipe.example?raw';
import spaceTimeExample from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/space-time-hot-spots-recipe.example?raw';
import type {TopicPageData} from '../../topic-types';
import drainageExample from './drainage-network.example?raw';

/** The declaration that starts with `start` through the next line that is exactly `};`. */
function cutDeclaration(source: string, start: string): string {
  const lines = source.split('\n');
  const first = lines.findIndex(line => line.startsWith(start));
  if (first < 0) return `// ${start} not found`;
  // Keep the doc comment above the declaration.
  let from = first;
  while (from > 0 && /^\s*(\/\*\*|\*)/.test(lines[from - 1])) from--;
  const last = lines.findIndex((line, index) => index > first && line === '};');
  return lines.slice(from, last < 0 ? first + 1 : last + 1).join('\n');
}

const recipeResultSource = [
  cutDeclaration(recipeUtilitiesSource, 'export type GPURecipeResult<'),
  cutDeclaration(statusSource, 'export type GPURecipeStatusPort ='),
  cutDeclaration(statusSource, 'export type GPURecipeStageStatusPort =')
].join('\n\n');

export const pageData: TopicPageData = {
  scenarios: [],
  examples: {
    'local-moran': localMoranExample,
    'hot-spots': hotSpotsExample,
    'rate-clusters': rateClustersExample,
    'space-time': spaceTimeExample,
    'drive-time': driveTimeExample,
    drainage: drainageExample,
    'recipe-result': recipeResultSource
  }
};
