// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Finds the live run of a catalog entry by id. A run lives in
// `entries/<group>/<entry-id>.ts` (the scene, imports luma.gl and the analysis operation) and,
// when it has sliders, in the sibling `entries/<group>/<entry-id>.parameters.ts` (tiny, no luma.gl
// imports). The glob below loads scenes lazily and parameter files eagerly, so prerendered pages can
// render the sliders without shipping any GPU code.

import type {GpuSceneLoader, GpuSceneModule, GpuSceneParameters} from '../types';
import type {RunScene} from './define-run';
import type {RunLegendItem} from './legend';

type SceneModule = {default: RunScene};

const SCENE_LOADERS = import.meta.glob<SceneModule>([
  './entries/*/*.ts',
  '!./entries/*/*.parameters.ts'
]);
const PARAMETER_FILES = import.meta.glob<GpuSceneParameters>('./entries/*/*.parameters.ts', {
  eager: true,
  import: 'parameters'
});

const LEGEND_FILES = import.meta.glob<{legend?: readonly RunLegendItem[]}>(
  './entries/*/*.parameters.ts',
  {eager: true}
);

function getEntryId(path: string): string {
  const fileName = path.slice(path.lastIndexOf('/') + 1);
  return fileName.replace(/(\.parameters)?\.ts$/, '');
}

const SCENE_PATHS = new Map<string, string>();
for (const path of Object.keys(SCENE_LOADERS)) SCENE_PATHS.set(getEntryId(path), path);
const PARAMETERS = new Map<string, GpuSceneParameters>();
for (const [path, parameters] of Object.entries(PARAMETER_FILES)) {
  PARAMETERS.set(getEntryId(path), parameters);
}

/** Color key of the entry's picture, when its run declares one. */
export function getRunLegend(entryId: string): readonly RunLegendItem[] | undefined {
  for (const [path, module] of Object.entries(LEGEND_FILES)) {
    if (module.legend && getEntryId(path) === entryId) return module.legend;
  }
  return undefined;
}

/** True when the catalog entry has a live run of its own operation. */
export function hasRun(entryId: string): boolean {
  return SCENE_PATHS.has(entryId);
}

/** Ids of every entry that has a run. */
export function listRunIds(): string[] {
  return [...SCENE_PATHS.keys()].sort();
}

/** The entry's group, which is the folder name under `entries/`. */
export function getRunGroup(entryId: string): string | undefined {
  return SCENE_PATHS.get(entryId)?.split('/')[2];
}

/** Slider parameters of the entry's run, available without loading the scene. */
export function getRunParameters(entryId: string): GpuSceneParameters | undefined {
  return hasRun(entryId) ? (PARAMETERS.get(entryId) ?? []) : undefined;
}

/** Scene loader for the entry's run, for `<GpuCanvas scene={...}>`; the module is lazy. */
export function getRunScene(entryId: string): GpuSceneLoader | undefined {
  const path = SCENE_PATHS.get(entryId);
  if (!path) return undefined;
  return async (): Promise<GpuSceneModule> => {
    const module = await SCENE_LOADERS[path]();
    return {default: module.default, parameters: [...(PARAMETERS.get(entryId) ?? [])]};
  };
}
