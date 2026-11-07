// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Suspends until a lens view model is loaded. The promise cache is module level, so prerendering
// resolves each model once and the client re-renders without a flash.

import {use} from 'react';
import {getLensViewModel} from './index.ts';
import type {LensViewModel} from './types.ts';

const cache = new Map<string, Promise<LensViewModel | undefined>>();

export function useLensViewModel(scenarioId: string): LensViewModel | undefined {
  let promise = cache.get(scenarioId);
  if (!promise) {
    promise = getLensViewModel(scenarioId);
    cache.set(scenarioId, promise);
  }
  return use(promise);
}
