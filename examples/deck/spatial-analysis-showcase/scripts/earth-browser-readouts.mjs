// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile} from 'node:fs/promises';

/** Prefer stable shell identifiers; retain exact-label fallback for older production snapshots. */
export async function installReadoutIdentifiers(page, sceneId, chapter = 'raster') {
  const source = await readFile(new URL(`../src/scenes/${chapter}/${sceneId}.scene.ts`, import.meta.url), 'utf8');
  const labels = Object.fromEntries([...source.matchAll(/\{\s*id:\s*'([^']+)',\s*label:\s*'([^']+)'/g)].map(match => [match[2], match[1]]));
  await page.evaluate(readoutLabels => {
    for (const row of document.querySelectorAll('.readout')) {
      const label = row.querySelector('dt')?.textContent;
      const value = row.querySelector('dd');
      const identifier = row.getAttribute('data-readout-id') ?? (label ? readoutLabels[label] : null);
      if (value && identifier) value.setAttribute('data-readout', identifier);
    }
  }, labels);
}
