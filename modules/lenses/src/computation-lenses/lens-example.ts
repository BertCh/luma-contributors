// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {publicEntryMaps} from './public-entry-maps';

function hasPublicEntry(name: string, packageName: string, entries: string[]): boolean {
  if (name !== packageName && !name.startsWith(`${packageName}/`)) return false;
  const subpath = name === packageName ? '.' : `.${name.slice(packageName.length)}`;
  if (
    subpath !== '.' &&
    subpath
      .slice(2)
      .split('/')
      .some(segment => !segment || segment === '.' || segment === '..')
  )
    return false;
  return entries.includes(subpath);
}

/** Maximum number of source lines shown beneath a result figure. */
export const LENS_EXAMPLE_MAX_LINES = 25;

/** Checked source snippet for a scenario's actual execution code. */
export type LensExample = {code: string; language: 'ts'; lineCount: number};

/** Extracts and dedents the unique example region, rejecting missing, empty or oversized blocks. */
export function extractLensExample(sourceText: string): LensExample {
  const lines = sourceText.replace(/\r\n?/g, '\n').split('\n');
  const starts = lines.flatMap((line, index) =>
    /^\s*\/\/\s*#region example\s*$/.test(line) ? [index] : []
  );
  const ends = lines.flatMap((line, index) =>
    /^\s*\/\/\s*#endregion example\s*$/.test(line) ? [index] : []
  );
  if (!starts.length || !ends.length) throw new Error('Missing example region markers');
  if (starts.length !== 1 || ends.length !== 1) throw new Error('Duplicated example region');
  if (ends[0] <= starts[0]) throw new Error('Example region end precedes its start');
  const snippet = lines.slice(starts[0] + 1, ends[0]);
  if (!snippet.length || snippet.every(line => !line.trim()))
    throw new Error('Empty example region');
  if (!snippet[0].trim() || !snippet[snippet.length - 1].trim())
    throw new Error('Example region has blank first or last line');
  if (snippet.length > LENS_EXAMPLE_MAX_LINES)
    throw new Error(`Example region exceeds ${LENS_EXAMPLE_MAX_LINES} lines (${snippet.length})`);
  const indent = Math.min(
    ...snippet.filter(line => line.trim()).map(line => line.match(/^\s*/)![0].length)
  );
  return {
    code: snippet.map(line => line.slice(indent)).join('\n'),
    language: 'ts',
    lineCount: snippet.length
  };
}

/** Lists imports outside the published @luma.gl core, engine and graph entry points and the analysis and lenses entry points. */
export function findNonPublicImports(sourceText: string): string[] {
  // Tokenize comments and strings first so commented imports and quoted code are ignored.
  const tokens = [
    ...sourceText.matchAll(
      /\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[A-Za-z_$][\w$]*|[^\s]/g
    )
  ]
    .map(match => match[0])
    .filter(token => !token.startsWith('//') && !token.startsWith('/*'));
  const imports: string[] = [];
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    let specifier: string | undefined;
    const call = (token === 'import' || token === 'require') && tokens[index + 1] === '(';
    if (token === 'from' || token === 'import') specifier = tokens[index + 1];
    if (call) specifier = tokens[index + 2];
    if (
      call &&
      (!specifier || !/^["'`]/.test(specifier) || ![')', ','].includes(tokens[index + 3]))
    ) {
      const prefix = specifier && /^["'`]/.test(specifier) ? `${specifier.slice(1, -1)} (` : '';
      imports.push(`${prefix}module specifier must be a single literal${prefix ? ')' : ''}`);
      continue;
    }
    if (!specifier || !/^["'`]/.test(specifier)) continue;
    const name = specifier.slice(1, -1);
    const publicEntry = publicEntryMaps.some(entry =>
      hasPublicEntry(name, entry.name, entry[token === 'require' ? 'require' : 'import'])
    );
    if (!publicEntry || /\/(?:src|dist|test|_internal)(?:\/|$)/.test(name)) imports.push(name);
  }
  return [...new Set(imports)];
}

/** Returns the sibling example path relative to the computation-lenses source directory. */
export function getLensExamplePath(scenarioId: string): string {
  if (
    !/^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)?$/.test(
      scenarioId
    )
  ) {
    throw new Error(`Invalid scenario id: ${scenarioId}`);
  }
  const [entry, element] = scenarioId.split('.');
  return `scenarios/${entry}/${element}.example.ts`;
}
