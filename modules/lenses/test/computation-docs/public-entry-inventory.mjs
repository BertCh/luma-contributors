// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, readdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {repositoryRoot} from './public-surface-census.mjs';

function filesBelow(root, prefix = '') {
  return readdirSync(resolve(root, prefix), {withFileTypes: true}).flatMap(entry => {
    const path = prefix + entry.name;
    return entry.isDirectory() ? filesBelow(root, path + '/') : [path];
  });
}

/** Installed luma.gl public entries are checked independently of the browser-safe inventory. */
export function createLumaEntryInventory() {
  return ['core', 'engine', 'graph'].map(directory => {
    const root = resolve(repositoryRoot, 'node_modules/@luma.gl', directory);
    const manifest = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
    return {
      name: manifest.name,
      ...Object.fromEntries(['import', 'require'].map(condition => [
        condition,
        Object.entries(manifest.exports)
          .filter(([entry, target]) =>
            !entry.split('/').includes('_internal') && selectRuntimeTarget(target, condition)
          )
          .map(([entry]) => entry)
          .sort()
      ]))
    };
  });
}

export function selectRuntimeTarget(target, condition) {
  if (typeof target === 'string' || target === null) return target;
  for (const [key, value] of Object.entries(target)) {
    if (key === condition || key === 'default' || key === 'node') {
      const selected = selectRuntimeTarget(value, condition);
      if (selected !== undefined) return selected;
    }
  }
  return undefined;
}

/** Expand real shipped source targets, retaining Node's import/require distinction. */
export function createPublicEntryInventory() {
  return ['analysis', 'lenses'].map(directory => {
    const root = resolve(repositoryRoot, 'modules', directory);
    const manifest = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8'));
    const sources = filesBelow(resolve(root, 'src'));
    return {name: manifest.name, ...Object.fromEntries(['import', 'require'].map(condition => {
      const entries = [];
      for (const [entry, conditions] of Object.entries(manifest.exports)) {
        const target = selectRuntimeTarget(conditions, condition);
        if (!target) continue;
        if (!entry.includes('*')) {
          entries.push(entry);
          continue;
        }
        const [prefix, suffix] = target.split('*');
        for (const source of sources) {
          // A require target is emitted as .cjs, an import target as .js.
          const emitted = './dist/' + source.replace(/\.ts$/, condition === 'require' ? '.cjs' : '.js');
          if (!/\.(?:[cm]?js|json)$/.test(emitted)) continue;
          if (emitted.startsWith(prefix) && emitted.endsWith(suffix)) {
            const replacement = emitted.slice(prefix.length, suffix ? -suffix.length : undefined);
            entries.push(entry.replace('*', replacement));
          }
        }
      }
      return [condition, [...new Set(entries)].sort()];
    }))};
  }).concat(createLumaEntryInventory());
}
