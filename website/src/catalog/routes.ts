// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Node-friendly list of catalog pages to prerender. vite.config.ts runs in Node without
// import.meta.glob, so entry ids are read from the entry source files with a tiny scan.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {DATA_TYPES, DOMAINS, USE_CASES} from './taxonomy';

const entriesDirectory = path.join(path.dirname(fileURLToPath(import.meta.url)), 'entries');

/** Entry ids declared in src/catalog/entries/*.ts (entries sit at a four-space indent, which example code inside strings does not match). */
export function listCatalogEntryIds(directory: string = entriesDirectory): string[] {
  if (!fs.existsSync(directory)) {
    return [];
  }
  const ids = new Set<string>();
  for (const fileName of fs.readdirSync(directory).filter(name => name.endsWith('.ts'))) {
    const source = fs.readFileSync(path.join(directory, fileName), 'utf8');
    for (const match of source.matchAll(/^ {4}id: '([a-z0-9-]+)',$/gm)) {
      ids.add(match[1]);
    }
  }
  return [...ids].sort();
}

/** `/explore`, every facet landing page and every `/fn/<id>` page. */
export function listCatalogPagePaths(directory?: string): string[] {
  return [
    '/explore',
    ...DATA_TYPES.map(facet => `/explore/data/${facet.id}`),
    ...USE_CASES.map(facet => `/explore/use/${facet.id}`),
    ...DOMAINS.map(facet => `/explore/domain/${facet.id}`),
    ...listCatalogEntryIds(directory).map(id => `/fn/${id}`)
  ];
}
