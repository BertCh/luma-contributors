// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// `virtual:catalog-summaries`: the lightweight part of every catalog entry (no explanation, params,
// outputs or example), evaluated at build time. Cards, filters and counts use it, so the full
// entries (about 1 MB of source) load only on function pages, one group chunk at a time.

import fs from 'node:fs';
import path from 'node:path';
import {transform} from 'esbuild';
import type {Plugin} from 'vite';
import {attachCatalogAnalysisMetadata} from '../src/catalog/analysis-metadata';

const SUMMARIES_ID = 'virtual:catalog-summaries';
const RESOLVED_SUMMARIES_ID = `\0${SUMMARIES_ID}`;

/** Fields kept out of the summaries; they are only needed on the function page. */
const DETAIL_FIELDS = [
  'explanation',
  'params',
  'moreRequired',
  'outputs',
  'example',
  'interpretation'
] as const;

export function catalogPlugin({entriesDirectory}: {entriesDirectory: string}): Plugin {
  return {
    name: 'luma-contributors-catalog',

    resolveId(id) {
      return id === SUMMARIES_ID ? RESOLVED_SUMMARIES_ID : null;
    },

    async load(id) {
      if (id !== RESOLVED_SUMMARIES_ID) {
        return null;
      }
      const summaries: Record<string, unknown>[] = [];
      for (const fileName of fs
        .readdirSync(entriesDirectory)
        .filter(name => name.endsWith('.ts'))) {
        const filePath = path.join(entriesDirectory, fileName);
        this.addWatchFile(filePath);
        const group = fileName.replace(/\.ts$/, '');
        for (const entry of await loadEntries(filePath)) {
          const summary: Record<string, unknown> = {
            ...attachCatalogAnalysisMetadata(entry as {symbol: string}),
            group
          };
          for (const field of DETAIL_FIELDS) {
            delete summary[field];
          }
          summaries.push(summary);
        }
      }
      return `export default ${JSON.stringify(summaries)};`;
    }
  };
}

/** Entry files only have type imports, so stripping types is enough to evaluate them. */
async function loadEntries(filePath: string): Promise<Record<string, unknown>[]> {
  const source = fs.readFileSync(filePath, 'utf8');
  const {code} = await transform(source, {loader: 'ts', format: 'esm'});
  const moduleUrl = `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`;
  const moduleExports = (await import(moduleUrl)) as {entries?: Record<string, unknown>[]};
  return moduleExports.entries ?? [];
}
