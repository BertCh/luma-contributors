// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Writes upstream-data.json from the repository. Run from website/:
//   node --experimental-strip-types src/components/meta/pages/upstream-first/generate-data.ts
// Reads: package.json (the tested luma.gl pin), modules/analysis/package.json (entry points), modules/analysis/src/**
// (imports of @luma.gl by entry point, source files, contributor classes) and
// dev-docs/upstream-workarounds.md (one row per capability, by Status column).

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '../../../../../..');
const analysisSource = path.join(root, 'modules/analysis/src');

const packageJson = JSON.parse(
  fs.readFileSync(path.join(root, 'modules/analysis/package.json'), 'utf8')
) as {peerDependencies: Record<string, string>; exports: Record<string, unknown>};
// The version shown is the exact luma.gl pin this workspace is tested against (the root
// development dependency), not the peer range the analysis package accepts.
const workspaceManifest = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')) as {
  devDependencies: Record<string, string>;
};
const lumaVersion = workspaceManifest.devDependencies['@luma.gl/graph'];
const entryPoints = Object.keys(packageJson.exports).length;

function listSourceFiles(folder: string): string[] {
  return fs.readdirSync(folder, {withFileTypes: true}).flatMap(entry => {
    const full = path.join(folder, entry.name);
    if (entry.isDirectory()) return listSourceFiles(full);
    return /\.(ts|tsx)$/.test(entry.name) && !entry.name.endsWith('.d.ts') ? [full] : [];
  });
}

// An import or re-export statement: everything up to `from '<specifier>'` or a bare `import '<specifier>'`.
const IMPORT_PATTERN =
  /(?:^|\n)\s*(?:import|export)\s+(?:type\s+)?(?:[^;'"]*?\sfrom\s*)?['"](@luma\.gl\/[^'"]+)['"]/g;
const CONTRIBUTOR_PATTERN = /export class (\w+)[^{]*implements[^{]*GPUCommandNodeProducer/g;

const files = listSourceFiles(analysisSource);
const importCounts = new Map<string, number>();
const contributors = new Set<string>();
for (const file of files) {
  const text = fs.readFileSync(file, 'utf8');
  for (const match of text.matchAll(IMPORT_PATTERN)) {
    importCounts.set(match[1], (importCounts.get(match[1]) ?? 0) + 1);
  }
  for (const match of text.matchAll(CONTRIBUTOR_PATTERN)) contributors.add(match[1]);
}
const imports = [...importCounts.entries()]
  .map(([specifier, count]) => ({specifier, count}))
  .sort((left, right) => right.count - left.count || left.specifier.localeCompare(right.specifier));

// Workarounds register: table rows whose Status column starts with a known word.
const register = fs.readFileSync(path.join(root, 'dev-docs/upstream-workarounds.md'), 'utf8');
const STATUS_WORDS = ['dropped', 'workaround', 'accepted regression', 'obsolete', 'unmeasured'];
const statusCounts: Record<string, number> = Object.fromEntries(
  STATUS_WORDS.map(word => [word, 0])
);
for (const line of register.split('\n')) {
  if (!line.startsWith('| ')) continue;
  // Split on unescaped pipes; the status is the last cell.
  const cells = line
    .split(/(?<!\\)\|/)
    .map(cell => cell.trim())
    .filter(Boolean);
  const status = cells.at(-1)?.toLowerCase() ?? '';
  const word = STATUS_WORDS.find(candidate => status.startsWith(candidate));
  if (word) statusCounts[word]++;
}

const data = {
  lumaVersion,
  entryPoints,
  sourceFileCount: files.length,
  contributorClassCount: contributors.size,
  importStatementCount: imports.reduce((total, entry) => total + entry.count, 0),
  imports,
  workarounds: statusCounts
};
fs.writeFileSync(path.join(directory, 'upstream-data.json'), `${JSON.stringify(data, null, 2)}\n`);
process.stdout.write(`wrote ${imports.length} entry points\n`);
