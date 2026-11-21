// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';

import {getPublicPackageSpecifierIssue} from './doc-program';

const DOCUMENT_ROOTS = ['docs'];
const DOCUMENT_FILES = ['README.md', 'modules/analysis/README.md'];
const SPECIFIER = /@luma-contributors\/analysis((?:\/[A-Za-z0-9_.-]+)*)/g;

function listDocumentFiles(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return listDocumentFiles(path);
    return /\.(md|mdx|json)$/.test(entry.name) ? [path] : [];
  });
}

/** Every `@luma-contributors/analysis...` specifier in the docs, with the place it occurs. */
function collectDocumentedSpecifiers(): {specifier: string; subpath: string; location: string}[] {
  const files = [...DOCUMENT_ROOTS.flatMap(listDocumentFiles), ...DOCUMENT_FILES];
  return files.flatMap(file =>
    readFileSync(file, 'utf8')
      .split('\n')
      .flatMap((line, lineIndex) =>
        [...line.matchAll(SPECIFIER)].map(match => {
          // A sentence-final period is prose punctuation. Quotes/backticks immediately before
          // the token identify a literal specifier, whose full suffix must remain intact.
          const isLiteral = /['"`]/.test(line[(match.index ?? 0) - 1] ?? '');
          const specifier = isLiteral ? match[0] : match[0].replace(/\.$/, '');
          return {
            specifier,
            subpath: `.${specifier.slice('@luma-contributors/analysis'.length)}`,
            location: `${file}:${lineIndex + 1}`
          };
        })
      )
  );
}

it('documents only analysis package paths that are keys of the exports map', () => {
  const manifest = JSON.parse(readFileSync('modules/analysis/package.json', 'utf8')) as {
    exports: Record<string, unknown>;
  };
  const exported = new Set(Object.keys(manifest.exports));
  const specifiers = collectDocumentedSpecifiers();
  // The raster and dataframe entries are the ones the cross-reference once misnamed.
  expect(specifiers.some(entry => entry.subpath === './raster-analysis')).toBe(true);
  expect(specifiers.some(entry => entry.subpath === './dataframe-analysis')).toBe(true);
  const unexported = specifiers
    .filter(entry => !exported.has(entry.subpath))
    .map(entry => `${entry.location} ${entry.specifier}`);
  expect(unexported).toEqual([]);
});

it('rejects the retired gpu-raster and gpu-dataframe analysis subpaths', () => {
  // Discriminating control: the two names the cross-reference used are not exported, so a
  // documented `@luma-contributors/analysis/gpu-raster` would fail the check above.
  const manifest = JSON.parse(readFileSync('modules/analysis/package.json', 'utf8')) as {
    exports: Record<string, unknown>;
  };
  expect(Object.hasOwn(manifest.exports, './gpu-raster')).toBe(false);
  expect(Object.hasOwn(manifest.exports, './gpu-dataframe')).toBe(false);
  const line = 'import {GPUIsolines} from `@luma-contributors/analysis/gpu-raster`.';
  expect([...line.matchAll(SPECIFIER)].map(match => `.${match[1]}`)).toEqual(['./gpu-raster']);
});

it('DOC-110 rejects a real source leaf before repository paths resolve it', () => {
  expect(
    getPublicPackageSpecifierIssue('@luma-contributors/analysis/analysis-core')
  ).toBeUndefined();
  const privateLeaf = '@luma-contributors/analysis/analysis-core/kernels/gpu-contributor-utils';
  expect(existsSync('modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts')).toBe(
    true
  );
  expect(getPublicPackageSpecifierIssue(privateLeaf)).toBe(
    `unpublished package specifier ${privateLeaf}`
  );
});

it('REVIEW earlier-fix scanner retains dotted suffixes in prose and opted-out fences', () => {
  for (const text of [
    '`@luma-contributors/analysis/gpu-terrain.js`',
    "{/* no-typecheck: illustrative import */}\n```ts\nrequire('@luma-contributors/analysis/gpu-terrain.js');\n```"
  ]) {
    const matches = [...text.matchAll(SPECIFIER)];
    expect(matches.map(match => match[0])).toEqual(['@luma-contributors/analysis/gpu-terrain.js']);
    expect(getPublicPackageSpecifierIssue(matches[0][0])).toContain(
      'unpublished package specifier'
    );
  }
});
