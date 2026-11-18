// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {
  ANALYSIS_DOC_PAGES,
  DOCS_ROOT,
  REPOSITORY_ROOT,
  createVirtualProgram,
  formatDiagnostic,
  readDocPage
} from './doc-program';

const PAGE = 'operation-forms-and-cpu-twins.md';

type ContentsItem = string | {items?: ContentsItem[]};

function getContentsPages(items: ContentsItem[]): string[] {
  return items.flatMap(item =>
    typeof item === 'string' ? [item] : getContentsPages(item.items ?? [])
  );
}

it('the operation guide is reachable from the API index and table of contents', () => {
  expect(readDocPage('README.md')).toContain('/docs/api/operation-forms-and-cpu-twins');
  const contents: ContentsItem[] = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
  );
  expect(getContentsPages(contents)).toContain(`api/${PAGE.replace(/\.md$/, '')}`);
  expect(ANALYSIS_DOC_PAGES).toContain(PAGE);
});

it('the complete variogram walkthrough typechecks without undeclared example inputs', () => {
  // These fences are successive steps of one example. Unlike fragment checks, this program
  // supplies no implicit imports or `any` declarations: every API and bin must be declared.
  const variogramSection = readDocPage(PAGE).split('## Point density from device to readback')[0];
  const snippets = [...variogramSection.matchAll(/^```ts\n([\s\S]*?)^```/gm)];
  expect(snippets.length).toBe(2);
  const path = join(DOCS_ROOT, '__doc_snippets__', 'operation-forms-and-cpu-twins.ts');
  const program = createVirtualProgram(
    new Map([[path, snippets.map(match => match[1]).join('\n')]])
  );
  const source = program.getSourceFile(path);
  expect(source).toBeDefined();
  const diagnostics = [
    ...program.getSyntacticDiagnostics(source),
    ...program.getSemanticDiagnostics(source)
  ];
  expect(diagnostics.map(formatDiagnostic)).toEqual([]);
}, 120_000);
