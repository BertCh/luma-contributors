// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Lens view models: compact, build-time extracts of docs/lens-records/*.json (see
// generate-view-models.ts). Safe to call from route loaders on the server and in the browser; each
// model is its own lazily loaded chunk, so the record files (about 29 MB) never ship.

import type {LensViewModel} from './types.ts';

export type {LensViewModel, LensViewNode, LensViewEdge, LensMap, LensSeries} from './types.ts';

const loaders = import.meta.glob<{default: LensViewModel}>('./generated/*.json');

const PATH_PATTERN = /\.\/generated\/(.+)\.json$/;

/** Scenario ids that have a view model, sorted. */
export function listLensViewModelIds(): string[] {
  return Object.keys(loaders)
    .map(path => PATH_PATTERN.exec(path)?.[1] ?? '')
    .filter(Boolean)
    .sort();
}

export function hasLensViewModel(scenarioId: string): boolean {
  return `./generated/${scenarioId}.json` in loaders;
}

/** Loads the view model of a scenario id such as `gpu-spatial-analysis.kernel-density`. */
export async function getLensViewModel(scenarioId: string): Promise<LensViewModel | undefined> {
  const load = loaders[`./generated/${scenarioId}.json`];
  return load ? (await load()).default : undefined;
}

export {FEATURED_LENS_SCENARIOS} from './featured.ts';
