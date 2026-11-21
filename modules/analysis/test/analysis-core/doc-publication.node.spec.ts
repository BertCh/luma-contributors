// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync, readdirSync} from 'node:fs';
import {join, posix, relative, resolve} from 'node:path';
import {expect, it} from 'vitest';
import type {Plugin} from 'vite';
import {decodeHTML, decodeHTMLAttribute} from 'entities';
import {listCatalogPagePaths} from '../../../../website/src/catalog/routes';
import {
  ANALYSIS_DOC_PAGES,
  DOCS_ROOT,
  REPOSITORY_ROOT,
  getSidebarPageIssues,
  discoverApiDocPages
} from './doc-program';

import {
  getDocSections,
  getRenderedDocDestinations,
  getRenderedDocTableRows
} from './doc-heading-sections.test-utils';

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

it('keeps old guide paths as link-only compatibility pointers to substantive published guides', async () => {
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
    expect(await getMigratedGuideLinkIssues(page, text)).toEqual([]);
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

function getPublishedDocRoute(page: string): string {
  const slug = posix
    .normalize(`api/${page}`)
    .replace(/\.md$/, '')
    .replace(/(^|\/)README$/, '');
  return `/docs/${slug}`.replace(/\/$/, '');
}

/** Same URL rewrite as the site renderer, pinned below against its actual output. */
function getRenderedPublishedHref(page: string, href: string): string {
  if (/^[a-z]+:/i.test(href) || href.startsWith('#') || href.startsWith('//')) return href;
  const [target, fragment] = href.split('#');
  if (!target.endsWith('.md') && !target.startsWith('/docs/')) return href;
  const docsPath = target.startsWith('/docs/')
    ? target.slice('/docs/'.length)
    : posix.normalize(posix.join(posix.dirname(`api/${page}`), target));
  const slug = docsPath.replace(/\.md$/, '').replace(/(^|\/)README$/, '');
  return `/docs/${slug}${fragment ? `#${fragment}` : ''}`;
}

let publicationPages: Map<string, string> | undefined;
function getPublicationPages(): Map<string, string> {
  publicationPages ??= new Map(
    discoverApiDocPages(resolve(DOCS_ROOT, '..')).map(page => {
      const path = resolve(DOCS_ROOT, '..', page);
      return [relative(DOCS_ROOT, path), readFileSync(path, 'utf8')];
    })
  );
  return publicationPages;
}

function getPublicSitePaths(
  directory = join(REPOSITORY_ROOT, 'website/public'),
  prefix = ''
): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry =>
    entry.isDirectory()
      ? getPublicSitePaths(join(directory, entry.name), `${prefix}/${entry.name}`)
      : entry.isFile()
        ? [`${prefix}/${entry.name}`]
        : []
  );
}
const PUBLIC_SITE_PATHS = new Set(getPublicSitePaths());
// Static application routes come from their route declarations. Function routes additionally
// require an actual entry ID; the docs splat must never certify a missing Markdown page.
const APPLICATION_ROUTES = new Set(
  getPublicSitePaths(join(REPOSITORY_ROOT, 'website/src/routes'))
    .filter(path => path.endsWith('.tsx'))
    .flatMap(path => {
      const source = readFileSync(join(REPOSITORY_ROOT, 'website/src/routes', path), 'utf8');
      return [...source.matchAll(/createFileRoute\('([^']+)'\)/g)]
        .map(match => match[1].replace(/\/$/, ''))
        .filter(route => !route.includes('$'));
    })
);
for (const path of listCatalogPagePaths()) APPLICATION_ROUTES.add(path);
// Lens recording pages (/lens/<scenario>) are prerendered from the same listing the site build uses.
const {listLensRecordingPagePaths} = await import('../../../../website/build/markdown-plugin');
for (const path of await listLensRecordingPagePaths())
  APPLICATION_ROUTES.add(path.replace(/\/$/, ''));

async function getApiLinkIssues(
  page: string,
  markdown: string,
  pages: ReadonlyMap<string, string>,
  publicPaths: ReadonlySet<string> = PUBLIC_SITE_PATHS
): Promise<string[]> {
  const rendered = await getActualPublishedDestinations(page, markdown);
  return getEmittedLinkIssues(page, rendered.links, pages, publicPaths);
}

/** Resolve the URLs actually emitted by the publication transform, after browser normalization. */
async function getEmittedLinkIssues(
  page: string,
  links: readonly {href: string; line: number}[],
  pages: ReadonlyMap<string, string>,
  publicPaths: ReadonlySet<string> = PUBLIC_SITE_PATHS
): Promise<string[]> {
  const routes = new Map(
    [...pages].map(([target, text]) => [getPublishedDocRoute(target), {page: target, text}])
  );
  const origin = 'https://published-docs.example';
  const issues: string[] = [];
  for (const {href, line} of links) {
    if (/^[a-z][a-z\d+.-]*:/i.test(href) || href.startsWith('//')) continue;
    const url = new URL(href, `${origin}${getPublishedDocRoute(page)}`);
    const pathname = decodeURIComponent(url.pathname).replace(/\/$/, '');
    const target = routes.get(pathname);
    if (target === undefined) {
      if (publicPaths.has(decodeURIComponent(url.pathname)) || APPLICATION_ROUTES.has(pathname))
        continue;
      issues.push(`${page}:${line} missing published destination ${href} -> ${url.pathname}`);
      continue;
    }
    const fragment = decodeURIComponent(url.hash.slice(1));
    if (
      fragment &&
      !(await getActualPublishedDestinations(target.page, target.text)).anchors.includes(fragment)
    )
      issues.push(`${page}:${line} missing rendered API fragment ${href}`);
  }
  return issues;
}

it('DOC-107 validates absolute, relative and same-page rendered fragments including duplicate linked headings', async () => {
  const target = '# [Overview](https://example.com)\n\n## Overview\n\n## `GPUExample`\n';
  const markdown =
    '# Start\n\n[absolute](/docs/api/target#overview-1)\n[relative](target.md#gpuexample)\n[same](#start)\n';
  const pages = new Map([
    ['source.md', markdown],
    ['target.md', target]
  ]);
  expect(await getApiLinkIssues('source.md', markdown, pages)).toEqual([]);
  expect(getRenderedDocDestinations(target).anchors).toEqual([
    'overview',
    'overview-1',
    'gpuexample'
  ]);
  for (const [valid, invalid] of [
    ['#overview-1', '#overview-2'],
    ['#gpuexample', '#missing'],
    ['#start', '#gone']
  ]) {
    const issues = await getApiLinkIssues('source.md', markdown.replace(valid, invalid), pages);
    expect(issues).toHaveLength(1);
    expect(issues[0]).toContain(invalid);
  }
});

function parseTransformedHtml(result: unknown): string {
  if (
    !result ||
    typeof result !== 'object' ||
    !('code' in result) ||
    typeof result.code !== 'string'
  ) {
    throw new Error('Markdown transform must return code');
  }
  const {html} = JSON.parse(result.code.replace(/^export default /, '').replace(/;$/, '')) as {
    html: string;
  };
  return html;
}

it('round 1 B1 rejects absent and non-string transform code with the guarded diagnostic', () => {
  for (const result of [
    undefined,
    null,
    'plain code',
    {},
    {code: undefined},
    {code: {toString: () => 'export default {}'}}
  ]) {
    expect(() => parseTransformedHtml(result)).toThrow('Markdown transform must return code');
  }
  expect(parseTransformedHtml({code: 'export default {"html":"<h1>Valid</h1>"};'})).toBe(
    '<h1>Valid</h1>'
  );
});

it('round 1 B3 preserves emitted raw anchors and follows rendered HTML body links', async () => {
  const pages = new Map([
    ['README.md', '# API\n\n<a href="/docs/api/target#explicit-anchor">Raw link</a>'],
    ['target.md', '# Target\n\n<a id="explicit-anchor"></a>\n<span id=unquoted-anchor></span>']
  ]);
  expect(await getApiLinkIssues('README.md', pages.get('README.md')!, pages)).toEqual([]);
  expect(await getApiLinkIssues('README.md', '[second](target.md#unquoted-anchor)', pages)).toEqual(
    []
  );
  expect(await getBodyReachablePages('README.md', pages)).toEqual(new Set(pages.keys()));
});

it('round 1 B4 rejects example buffer IDs as routes and accepts registered facet pages', async () => {
  const pages = new Map([['README.md', '# API']]);
  expect(await getApiLinkIssues('README.md', '[buffer](/fn/moran-parameters)', pages)).toHaveLength(
    1
  );
  expect(await getApiLinkIssues('README.md', '[facet](/explore/data/points)', pages)).toEqual([]);
});

async function getActualPublishedHtml(page: string, markdown: string): Promise<string> {
  const compiler = (await import(join(REPOSITORY_ROOT, 'website/build/markdown-plugin.ts'))) as {
    markdownPlugin: (options: {docsDirectory: string}) => Plugin;
  };
  const plugin = compiler.markdownPlugin({docsDirectory: resolve(DOCS_ROOT, '..')});
  if (typeof plugin.transform !== 'function')
    throw new Error('Markdown transform must be callable');
  const result = await plugin.transform.call({} as never, markdown, join(DOCS_ROOT, page));
  return parseTransformedHtml(result);
}

async function getActualPublishedLinks(page: string, markdown: string): Promise<string[]> {
  return (await getActualPublishedDestinations(page, markdown)).links.map(link => link.href);
}

type PublishedDestinations = {
  anchors: string[];
  links: {href: string; label: string; line: number}[];
};
const publishedDestinations = new Map<string, Promise<PublishedDestinations>>();

function getActualPublishedDestinations(
  page: string,
  markdown: string
): Promise<PublishedDestinations> {
  const key = JSON.stringify([page, markdown]);
  let pending = publishedDestinations.get(key);
  if (!pending) {
    pending = getActualPublishedHtml(page, markdown).then(html => {
      const anchors: string[] = [];
      const links: PublishedDestinations['links'] = [];
      // The renderer emits raw HTML verbatim. Inspect its tags, including quoted/unquoted IDs,
      // instead of asking a Markdown parser to reconstruct an HTML document. Escaped code text
      // has no tags; comments and script/style contents must not create navigation edges.
      const body = html.replace(/<!--[\s\S]*?-->|<(script|style)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, '');
      for (const match of body.matchAll(/<([a-z][\w:-]*)\b((?:"[^"]*"|'[^']*'|[^'">])*)>/gi)) {
        const attributes = new Map<string, string>();
        for (const attribute of match[2].matchAll(
          /(?:^|\s)([\w:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g
        )) {
          attributes.set(
            attribute[1].toLowerCase(),
            decodeHTMLAttribute(attribute[2] ?? attribute[3] ?? attribute[4] ?? '')
          );
        }
        const id = attributes.get('id');
        if (id) anchors.push(id);
        if (match[1].toLowerCase() !== 'a') continue;
        const name = attributes.get('name');
        if (name) anchors.push(name);
        const href = attributes.get('href');
        if (href === undefined) continue;
        const after = (match.index ?? 0) + match[0].length;
        const end = body.slice(after).search(/<\/a\s*>/i);
        const label = decodeHTML(
          body.slice(after, end < 0 ? after : after + end).replace(/<[^>]*>/g, '')
        );
        links.push({href, label, line: body.slice(0, match.index).split('\n').length});
      }
      return {anchors, links};
    });
    publishedDestinations.set(key, pending);
  }
  return pending;
}

async function verifyEvidencePublication(renderLinks = getActualPublishedLinks): Promise<void> {
  const pages = getPublicationPages();
  for (const [page, evidence] of [
    ['analysis-catalog.md', 'examples/deck/kepler-compute/VALIDATION-CATALOG.md'],
    [
      'gpu-spatial-analysis/vecchia-likelihood.md',
      'dev-docs/research/spatiotemporal-review-2026-11-09/README.md'
    ],
    [
      'pointcloud-analysis.md',
      'examples/deck/kepler-compute/README.md#point-cloud-contributor-catalog'
    ]
  ]) {
    const links = await renderLinks(page, pages.get(page)!);
    const expected = `${REPOSITORY_FILE_BASE}${evidence}`;
    expect(links, page).toContain(expected);
    expect(
      await getEmittedLinkIssues(
        page,
        links.map(href => ({href, line: 0})),
        pages
      )
    ).toEqual([]);
    const [file, fragment] = evidence.split('#');
    expect(hasRepositoryFileLink(`${REPOSITORY_FILE_BASE}${file}`)).toBe(true);
    if (fragment)
      expect(
        getRenderedDocDestinations(readFileSync(join(REPOSITORY_ROOT, file), 'utf8')).anchors
      ).toContain(fragment);
  }
  const broken = '/docs/../examples/deck/kepler-compute/VALIDATION-CATALOG';
  expect(
    await getEmittedLinkIssues('analysis-catalog.md', [{href: broken, line: 1}], pages)
  ).toContainEqual(expect.stringContaining('-> /examples/deck/kepler-compute/VALIDATION-CATALOG'));
  // Normalization must not let dot segments, encoded segments or a real route with a bad fragment pass.
  for (const href of ['/docs/../missing', '/docs/%2e%2e/missing', '/docs/api#missing-fragment']) {
    expect(
      await getEmittedLinkIssues('analysis-catalog.md', [{href, line: 1}], pages)
    ).toHaveLength(1);
  }
}

it('DST-13 checks real emitted destinations and preserves the three repository evidence links', async () => {
  await verifyEvidencePublication();
});

it('round 1 S3 tolerates a corrected repository rewrite without requiring the old broken route', async () => {
  await verifyEvidencePublication(async (page, markdown) =>
    (await getActualPublishedLinks(page, markdown)).map(href =>
      href === '/docs/../examples/deck/kepler-compute/VALIDATION-CATALOG'
        ? `${REPOSITORY_FILE_BASE}examples/deck/kepler-compute/VALIDATION-CATALOG.md`
        : href
    )
  );
});

async function getBodyReachablePages(
  start: string,
  pages: ReadonlyMap<string, string>
): Promise<Set<string>> {
  const routes = new Map([...pages.keys()].map(page => [getPublishedDocRoute(page), page]));
  const reached = new Set<string>();
  const pending = [start];
  while (pending.length) {
    const page = pending.pop()!;
    if (reached.has(page) || !pages.has(page)) continue;
    reached.add(page);
    for (const {href} of (await getActualPublishedDestinations(page, pages.get(page)!)).links) {
      if (/^[a-z][a-z\d+.-]*:/i.test(href) || href.startsWith('//')) continue;
      const url = new URL(href, `https://published-docs.example${getPublishedDocRoute(page)}`);
      const destination = routes.get(decodeURIComponent(url.pathname).replace(/\/$/, ''));
      if (destination) pending.push(destination);
    }
  }
  return reached;
}

type PublicationContentsItem = string | {label?: string; items: PublicationContentsItem[]};

function getFamilyInventory(label: string, contents: readonly PublicationContentsItem[]): string[] {
  const collect = (items: readonly PublicationContentsItem[]): string[] =>
    items.flatMap(item =>
      typeof item === 'string' ? [item.replace(/^api\//, '') + '.md'] : collect(item.items)
    );
  for (const item of contents) {
    if (typeof item === 'string') continue;
    if (item.label === label) return collect(item.items);
    const found = getFamilyInventory(label, item.items);
    if (found.length) return found;
  }
  return [];
}

it('DST-12 independently traverses body links from each family rather than borrowing global reachability', async () => {
  const pages = new Map([
    ['README.md', '# API\n\n[A](a/README.md) [B](b/README.md)'],
    ['a/README.md', '# A\n\n[local](local.md)\n\n```md\n[hidden](orphan.md)\n```'],
    ['a/local.md', '# Local'],
    ['a/orphan.md', '# Only B links here'],
    [
      'b/README.md',
      '# B\n\n[other family](../a/orphan.md) [external](https://example.com/docs/api/a/absent)'
    ]
  ]);
  expect(await getBodyReachablePages('README.md', pages)).toEqual(new Set(pages.keys()));
  expect(await getBodyReachablePages('a/README.md', pages)).toEqual(
    new Set(['a/README.md', 'a/local.md'])
  );
  expect((await getBodyReachablePages('b/README.md', pages)).has('a/orphan.md')).toBe(true);
});

it('DST-12 reaches the API inventory through body links from the root and every family landing', async () => {
  const pages = getPublicationPages();
  // Sidebar records determine the expected inventory only; traversal never follows the sidebar.
  const contents = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
  ) as PublicationContentsItem[];
  const missing: string[] = [];
  for (const [start, label] of [
    ['README.md', ''],
    ['gpu-spatial-analysis/README.md', 'Spatial analysis'],
    ['gpu-network/README.md', 'Network'],
    ['gpu-raster/README.md', 'Raster analysis'],
    ['gpu-terrain/README.md', 'Terrain']
  ]) {
    const family = start === 'README.md' ? '' : start.slice(0, start.lastIndexOf('/') + 1);
    const inventory = label ? getFamilyInventory(label, contents) : [];
    if (label) expect(inventory.length, `${label} sidebar inventory`).toBeGreaterThan(0);
    const expected = new Set([
      ...ANALYSIS_DOC_PAGES.filter(page => page.startsWith(family)),
      ...inventory,
      ...(label === 'Spatial analysis' ? ['gpu-spatial-analysis-cross-reference.md'] : [])
    ]);
    const reached = await getBodyReachablePages(start, pages);
    missing.push(
      ...[...expected]
        .filter(page => !reached.has(page))
        .map(page => `${start} cannot reach ${page}`)
    );
  }
  expect(missing, missing.join('\n')).toEqual([]);
});

function getPreferredLinkIssues(
  page: string,
  entryPoint: string,
  links: readonly {href: string; label: string}[],
  targets: ReadonlyMap<string, string | undefined>
): string[] {
  return links.flatMap(link => {
    const identity = `${entryPoint}:${link.label}`;
    const target = targets.get(identity);
    if (!target) return [];
    const preferred = getRenderedPublishedHref('analysis-catalog.md', target);
    return link.href === preferred
      ? []
      : [`${page}: ${identity} -> ${link.href}; preferred ${preferred}`];
  });
}

it('DST-14 rejects wrong existing targets and accepts declared shared-family targets by qualified identity', () => {
  const entry = '@luma-contributors/analysis/gpu-spatial-analysis';
  const targets = new Map([
    [
      `${entry}:GPUSpatialPearson`,
      './gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic'
    ],
    [`${entry}:GPULocalGeary`, './gpu-spatial-analysis/operators-statistics.md#gpulocalgeary'],
    [
      `${entry}:GPUGammaStatistic`,
      './gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic'
    ],
    [
      `${entry}:GPUNearestFeatureWeights`,
      './gpu-spatial-analysis/operators-joins.md#gpunearestfeatureweights'
    ],
    [
      '@luma-contributors/analysis/raster-analysis:GPUExample',
      './gpu-raster/family.md#shared-family'
    ],
    ['@luma-contributors/analysis/gpu-network:GPUExample', './gpu-network/family.md#shared-family']
  ]);
  for (const [name, fragment] of [
    ['GPULocalGeary', 'gpuhotspotanalysis-and-gpulocalmoran'],
    ['GPUSpatialPearson', 'gpulocalpermutationtest-and-gpuglobalpermutationtest'],
    ['GPUGammaStatistic', 'additional-spatial-statistics-and-bounded-optimization'],
    ['GPUNearestFeatureWeights', 'gpupointinpolygonjoin-and-gpunearestfeaturejoin'],
    ['GPULocalGeary', 'additional-spatial-statistics-and-bounded-optimization'],
    ['GPUSpatialPearson', 'additional-spatial-statistics-and-bounded-optimization'],
    ['GPUGammaStatistic', 'additional-spatial-statistics-and-bounded-optimization']
  ]) {
    const page = `gpu-spatial-analysis/operators-${name === 'GPUNearestFeatureWeights' ? 'joins' : 'statistics'}.md`;
    expect(getRenderedDocDestinations(getPublicationPages().get(page)!).anchors).toContain(
      fragment
    );
    expect(
      getPreferredLinkIssues(
        'fixture.md',
        entry,
        [{label: name, href: `${getPublishedDocRoute(page)}#${fragment}`}],
        targets
      )
    ).toHaveLength(1);
  }
  for (const family of ['raster-analysis', 'gpu-network']) {
    const page = family === 'raster-analysis' ? 'gpu-raster' : 'gpu-network';
    expect(
      getPreferredLinkIssues(
        'fixture.md',
        `@luma-contributors/analysis/${family}`,
        [{label: 'GPUExample', href: `/docs/api/${page}/family#shared-family`}],
        targets
      )
    ).toEqual([]);
  }
});

const PREFERRED_CONTRACT_PINS = [
  // Independent navigation pins, reviewed against the named contract sections. These are
  // qualified public identities, even when a preferred shared heading names several classes.
  [
    '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalGeary',
    'gpu-spatial-analysis/operators-statistics.md#gpulocalgeary'
  ],
  [
    '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPearson',
    'gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic'
  ],
  [
    '@luma-contributors/analysis/gpu-spatial-analysis:GPUGammaStatistic',
    'gpu-spatial-analysis/operators-statistics.md#gpuspatialpearson-and-gpugammastatistic'
  ],
  [
    '@luma-contributors/analysis/gpu-spatial-analysis:GPUNearestFeatureWeights',
    'gpu-spatial-analysis/operators-joins.md#gpunearestfeatureweights'
  ]
];

function getCatalogPreferredTargets(pages: ReadonlyMap<string, string>): Map<string, string> {
  // Consume api-docs-catalog's shared qualified contract inventory rather than building a
  // second canonical-heading resolver. Literal pins below independently protect that inventory.
  const preferredTargets = new Map<string, string>();
  const rows = getRenderedDocTableRows(pages.get('analysis-catalog.md')!).filter(
    row =>
      row.headers[0] === 'Entry' &&
      row.headers[1] === 'Exports' &&
      row.sections.includes('Canonical prose contracts')
  );
  // These records explicitly declare preferred targets. The alphabetical browse index also
  // contains contextual links; it must not overwrite a canonical contract with a comparison.
  for (const row of rows) {
    const target = getRenderedDocDestinations(row.text).links[0]?.href;
    if (!target) continue;
    const entry =
      row.cells[0] === '.'
        ? '@luma-contributors/analysis'
        : `@luma-contributors/analysis/${row.cells[0]}`;
    for (const name of row.cells[1].match(/\bGPU[A-Z]\w+\b/g) ?? [])
      preferredTargets.set(`${entry}:${name}`, target);
  }
  return preferredTargets;
}

async function getPrimaryClassLinkIssues(
  page: string,
  markdown: string,
  pages: ReadonlyMap<string, string>
): Promise<string[]> {
  const preferredTargets = getCatalogPreferredTargets(pages);
  const issues: string[] = [];
  for (const link of (await getActualPublishedDestinations(page, markdown)).links) {
    if (!/^GPU[A-Z]\w+$/.test(link.label)) continue;
    const candidates = [...preferredTargets.keys()].filter(identity =>
      identity.endsWith(`:${link.label}`)
    );
    const local = `@luma-contributors/analysis/gpu-spatial-analysis:${link.label}`;
    const identity = preferredTargets.has(local)
      ? local
      : candidates.length === 1
        ? candidates[0]
        : undefined;
    if (!identity) {
      if (candidates.length) issues.push(`${page}: ambiguous public identity for ${link.label}`);
      continue;
    }
    issues.push(
      ...getPreferredLinkIssues(
        page,
        identity.slice(0, identity.lastIndexOf(':')),
        [link],
        preferredTargets
      )
    );
  }
  return issues;
}

it('DST-14 pins qualified Geary, Pearson, Gamma and nearest-feature links to their preferred contracts', async () => {
  const pages = getPublicationPages();
  const issues: string[] = [];
  const preferredTargets = getCatalogPreferredTargets(pages);
  for (const [identity, destination] of PREFERRED_CONTRACT_PINS) {
    expect(preferredTargets.get(identity), identity).toBe(`./${destination}`);
    const [targetPage, fragment] = destination.split('#');
    const section = getDocSections(pages.get(targetPage)!).find(
      candidate => candidate.anchor === fragment
    );
    expect(`${section?.heading}\n${section?.body}`, identity).toContain(
      identity.slice(identity.lastIndexOf(':') + 1)
    );
  }
  for (const page of [
    'gpu-spatial-analysis/operators.md',
    'gpu-spatial-analysis-cross-reference.md'
  ]) {
    issues.push(...(await getPrimaryClassLinkIssues(page, pages.get(page)!, pages)));
  }
  expect(issues, issues.join('\n')).toEqual([]);
});

it('round 1 B8 checks a preferred primary class outside the four literal pins', async () => {
  const pages = getPublicationPages();
  const page = 'gpu-spatial-analysis-cross-reference.md';
  const valid =
    '[`GPUSpatialWeightsAlgebra`](/docs/api/gpu-spatial-analysis/operators-weights#gpuspatialweightsalgebra)';
  expect(await getPrimaryClassLinkIssues(page, valid, pages)).toEqual([]);
  const broken = valid.replace('#gpuspatialweightsalgebra', '#gpuneighborsearch');
  expect(await getPrimaryClassLinkIssues(page, broken, pages)).toContainEqual(
    expect.stringContaining('GPUSpatialWeightsAlgebra')
  );
});

it('DOC-107 validates local rendered links across every API page and rejects the retired K-means fragment', async () => {
  const pages = getPublicationPages();
  const issues: string[] = [];
  for (const [page, markdown] of pages) {
    if (ANALYSIS_DOC_PAGES.includes(page))
      issues.push(...(await getApiLinkIssues(page, markdown, pages)));
  }
  expect(issues.length, issues.join('\n')).toBe(0);
}, 600_000);

it('DOC-107 rejects the retired K-means fragment on the existing statistics page', async () => {
  const pages = getPublicationPages();
  const page = 'gpu-spatial-analysis-cross-reference.md';
  const markdown = pages.get(page)!;
  expect(markdown).toContain('operators-statistics#gpuspatialkmeans');
  const broken = markdown.replace(
    'operators-statistics#gpuspatialkmeans',
    'operators-statistics#gpukmeans'
  );
  expect(await getApiLinkIssues(page, broken, pages)).toContainEqual(
    expect.stringContaining('operators-statistics#gpukmeans')
  );
});

it('REVIEW B2 rejects a nonexistent landing fragment and checkout-only repository links', async () => {
  const markdown = '# Landing\n\n[missing](/docs/api#invented-heading)';
  expect(
    await getApiLinkIssues('README.md', markdown, new Map([['README.md', markdown]]))
  ).toHaveLength(1);
  const repositoryLink = '[fixture](../../examples/deck/kepler-compute/VALIDATION-CATALOG.md)';
  expect(
    await getApiLinkIssues(
      'analysis-catalog.md',
      repositoryLink,
      new Map([['analysis-catalog.md', repositoryLink]])
    )
  ).toHaveLength(1);
});

function getMigratedGuideLinkIssues(page: string, markdown: string): Promise<string[]> {
  return getApiLinkIssues(page, markdown, getPublicationPages());
}

it('REVIEW S3 accepts a published absolute route from a migrated guide', async () => {
  expect(
    await getMigratedGuideLinkIssues(
      'analysis-core/fitting-kernels.md',
      '[statistics](/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialkmeans)'
    )
  ).toEqual([]);
});

it('REVIEW B2 pins URL rewriting to the actual renderer and resolves public routes', async () => {
  const page = 'analysis-catalog.md';
  const markdown =
    '# Fixture\n\n[landing](/docs/api#invented-heading)\n[relative](target.md#overview)';
  const links = await getActualPublishedLinks(page, markdown);
  expect(links).toEqual(
    getRenderedDocDestinations(markdown).links.map(link =>
      getRenderedPublishedHref(page, link.href)
    )
  );
  const landing = '# Landing';
  const pages = new Map([['README.md', landing]]);
  expect(await getApiLinkIssues('README.md', '[valid](/docs/api#landing)', pages)).toEqual([]);
  expect(
    await getApiLinkIssues('README.md', '[asset](/fixture.txt)', pages, new Set(['/fixture.txt']))
  ).toEqual([]);
  expect(
    await getApiLinkIssues('README.md', '[catalog](/explore/) [density](/fn/point-density)', pages)
  ).toEqual([]);
  expect(
    await getApiLinkIssues('README.md', '[missing](/fn/not-a-real-entry)', pages)
  ).toHaveLength(1);
});
