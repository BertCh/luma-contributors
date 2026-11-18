// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync, readdirSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {expect, it} from 'vitest';
import type {Plugin} from 'vite';
import {ANALYSIS_DOC_PAGES, DOCS_ROOT, REPOSITORY_ROOT, getSidebarPageIssues} from './doc-program';

const GUIDE_DESTINATIONS = {
  'chunked-spatial-weights-transpose': 'gpu-spatial-analysis/chunked-spatial-weights-transpose.md',
  'coregionalization-fit': 'dataframe-analysis/coregionalization-fit.md',
  'fitting-kernels': 'analysis-core/fitting-kernels.md',
  'geodesic-precision': 'gpu-spatial-analysis/geodesic-precision.md',
  'gpu-analysis-contiguity-statistics':
    'gpu-spatial-analysis/gpu-analysis-contiguity-statistics.md',
  'gpu-network-louvain': 'gpu-network/louvain.md',
  'gpu-quantile-lisa': 'gpu-spatial-analysis/quantile-lisa.md',
  'instrumental-variables-covariance': 'gpu-spatial-analysis/instrumental-variables-covariance.md',
  'isolines-tiles': 'gpu-raster/isolines-tiles.md',
  'terrain-dinf-hand': 'gpu-terrain/height-above-drainage.md'
};

const REPOSITORY_FILE_BASE = 'https://github.com/visgl/luma-contributors/blob/master/';

function hasRepositoryFileLink(link: string): boolean {
  if (!link.startsWith(REPOSITORY_FILE_BASE)) return false;
  const path = link.slice(REPOSITORY_FILE_BASE.length);
  return !path.split('/').includes('..') && existsSync(join(REPOSITORY_ROOT, path));
}

it('R7 rejects a missing fixture target or a different repository link base', () => {
  const fixture =
    'modules/analysis/test/gpu-dataframe/pair-statistics/fixtures/generate-coregionalization.py';
  expect(hasRepositoryFileLink(`${REPOSITORY_FILE_BASE}${fixture}`)).toBe(true);
  expect(hasRepositoryFileLink(`${REPOSITORY_FILE_BASE}${fixture}.missing`)).toBe(false);
  expect(
    hasRepositoryFileLink(`https://github.com/visgl/luma-contributors/blob/other/${fixture}`)
  ).toBe(false);
  expect(hasRepositoryFileLink(fixture)).toBe(false);
});

it('R2 renders the Kepler example link to an absolute repository URL', async () => {
  const compilerPath = join(REPOSITORY_ROOT, 'website/build/markdown-plugin.ts');
  const compiler = (await import(compilerPath)) as {
    markdownPlugin: (options: {docsDirectory: string}) => Plugin;
  };
  const plugin = compiler.markdownPlugin({docsDirectory: join(REPOSITORY_ROOT, 'docs')});
  if (typeof plugin.transform !== 'function')
    throw new Error('Markdown transform must be callable');
  const path = join(DOCS_ROOT, 'gpu-spatial-analysis/spatial-frames.md');
  const result = await plugin.transform.call({} as never, readFileSync(path, 'utf8'), path);
  if (!result || typeof result === 'string' || typeof result.code !== 'string') {
    throw new Error('Markdown transform must return code');
  }
  const rendered = JSON.parse(result.code.replace(/^export default /, '').replace(/;$/, '')) as {
    html: string;
  };
  const href = /href="([^"]+)"[^>]*>Kepler example<\/a>/.exec(rendered.html)?.[1];
  expect(href).toBe(
    'https://github.com/visgl/luma-contributors/blob/master/examples/deck/kepler-compute/README.md'
  );
});

it('publishes every formerly package-local guide through the shared API inventory and sidebar', () => {
  expect(
    Object.values(GUIDE_DESTINATIONS).filter(page => !ANALYSIS_DOC_PAGES.includes(page))
  ).toEqual([]);
  expect(getSidebarPageIssues()).toEqual([]);
});

it('keeps old guide paths as link-only compatibility pointers to substantive published guides', () => {
  for (const [name, page] of Object.entries(GUIDE_DESTINATIONS)) {
    const legacy = readFileSync(
      join(REPOSITORY_ROOT, 'modules/analysis/docs', `${name}.md`),
      'utf8'
    );
    expect(legacy, name).toContain(`../../../docs/api/${page}`);
    expect(legacy.length, `${name} must not duplicate the canonical contract`).toBeLessThan(400);
    const target = join(DOCS_ROOT, page);
    expect(existsSync(target), page).toBe(true);
    const text = readFileSync(target, 'utf8');
    expect(text.length, `${page} must preserve the guide, not only link back`).toBeGreaterThan(
      1000
    );
    for (const [, link] of text.matchAll(/\]\(([^\s)]+)\)/g)) {
      if (!/^(?:https?:|#)/.test(link)) {
        expect(existsSync(resolve(dirname(target), link.split('#')[0])), `${page}: ${link}`).toBe(
          true
        );
      }
    }
  }
});

it('uses published API paths in active catalog metadata and the authoring walkthrough', () => {
  const ontology = join(REPOSITORY_ROOT, 'modules/analysis/src/gpu-spatial-analysis/ontology');
  const sources = readdirSync(ontology, {recursive: true})
    .map(String)
    .filter(file => file.endsWith('.ts'));
  const stale = sources.filter(file =>
    readFileSync(join(ontology, file), 'utf8').includes('modules/analysis/docs/')
  );
  expect(stale).toEqual([]);
  for (const file of sources) {
    const text = readFileSync(join(ontology, file), 'utf8');
    for (const [, target] of text.matchAll(/documentation: '(docs\/api\/[^']+\.md)'/g)) {
      expect(existsSync(join(REPOSITORY_ROOT, target)), `${file}: ${target}`).toBe(true);
    }
  }
  expect(
    readFileSync(join(REPOSITORY_ROOT, 'dev-docs/authoring-walkthrough.md'), 'utf8')
  ).not.toContain('modules/analysis/docs/');
});

it('links installed-package readers to absolute guides and an existing fixture on the same branch', () => {
  const readme = readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/README.md'), 'utf8');
  for (const page of Object.values(GUIDE_DESTINATIONS)) {
    expect(readme).toContain(`${REPOSITORY_FILE_BASE}docs/api/${page}`);
  }
  const fixtureLink = /\[coregionalization fixture generator\]\(([^)]+)\)/.exec(readme)?.[1];
  expect(fixtureLink).toBe(
    `${REPOSITORY_FILE_BASE}modules/analysis/test/gpu-dataframe/pair-statistics/fixtures/generate-coregionalization.py`
  );
  expect(hasRepositoryFileLink(fixtureLink!), fixtureLink).toBe(true);
});
