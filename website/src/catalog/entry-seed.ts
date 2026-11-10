// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogEntry} from './schema';

/** Share the entry's deterministic input seed between the live scene and its poster. */
export function getEntrySeed(entry: Pick<CatalogEntry, 'id' | 'preview'>): number {
  if (entry.preview.seed !== undefined) {
    return entry.preview.seed;
  }
  let hash = 2166136261;
  for (let index = 0; index < entry.id.length; index++) {
    hash = Math.imul(hash ^ entry.id.charCodeAt(index), 16777619);
  }
  return (hash >>> 0) % 997;
}
