// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compiles docs/**/*.md to HTML at build time. Each markdown import becomes a module exporting
// {slug, title, html}, so the browser never parses markdown. `virtual:docs-index` lists every page.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import rehypeShiki from '@shikijs/rehype';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';
import remarkFrontmatter from 'remark-frontmatter';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import {unified} from 'unified';
import type {Plugin} from 'vite';
import {parse as parseYaml} from 'yaml';
import {loadEntries} from './catalog-plugin';

const DOCS_INDEX_ID = 'virtual:docs-index';
const RESOLVED_DOCS_INDEX_ID = `\0${DOCS_INDEX_ID}`;

export type DocsIndexEntry = {slug: string; title: string};

type HastNode = {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/** Lists every markdown file under the docs directory, relative and with forward slashes. */
export function listMarkdownFiles(docsDirectory: string): string[] {
  const files: string[] = [];
  const visit = (directory: string) => {
    for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(entryPath);
      } else if (entry.name.endsWith('.md')) {
        files.push(path.relative(docsDirectory, entryPath).split(path.sep).join('/'));
      }
    }
  };
  visit(docsDirectory);
  return files.sort();
}

/** Maps `api/gpu-network/README.md` to `api/gpu-network` and `api/geospatial.md` to `api/geospatial`. */
export function getSlugFromFile(relativeFile: string): string {
  return relativeFile.replace(/\.md$/, '').replace(/(^|\/)README$/, '');
}

/** Site paths of every docs page, for prerendering. */
export function listDocPagePaths(docsDirectory: string): string[] {
  return listMarkdownFiles(docsDirectory).map(file => `/docs/${getSlugFromFile(file)}`);
}

export function markdownPlugin({docsDirectory}: {docsDirectory: string}): Plugin {
  const resolvedDocsDirectory = path.resolve(docsDirectory);
  // The site may be served under a sub-path (Vite `base`); rendered links carry it.
  let basePath = '/';
  let lensDestinations: ReturnType<typeof loadLensDestinations> | undefined;

  return {
    name: 'luma-contributors-markdown',
    enforce: 'pre',

    configResolved(config) {
      basePath = config.base;
    },

    resolveId(id) {
      return id === DOCS_INDEX_ID ? RESOLVED_DOCS_INDEX_ID : null;
    },

    load(id) {
      if (id !== RESOLVED_DOCS_INDEX_ID) {
        return null;
      }
      const entries: DocsIndexEntry[] = listMarkdownFiles(resolvedDocsDirectory).map(file => {
        const source = fs.readFileSync(path.join(resolvedDocsDirectory, file), 'utf8');
        return {slug: getSlugFromFile(file), title: getTitle(source, file)};
      });
      return `export default ${JSON.stringify(entries)};`;
    },

    async transform(source, id) {
      const filePath = id.split('?')[0];
      if (!filePath.endsWith('.md') || !filePath.startsWith(resolvedDocsDirectory)) {
        return null;
      }
      const relativeFile = path.relative(resolvedDocsDirectory, filePath).split(path.sep).join('/');
      const slug = getSlugFromFile(relativeFile);
      let destinations: ReadonlyMap<string, LensDestination> = new Map();
      if (/^[ \t]*::lens\{/m.test(source)) {
        lensDestinations ??= loadLensDestinations();
        destinations = await lensDestinations;
      }
      const html = await renderMarkdown(source, relativeFile, basePath, destinations);
      const page = {slug, title: getTitle(source, relativeFile), html};
      return {code: `export default ${JSON.stringify(page)};`, map: null};
    }
  };
}

async function renderMarkdown(
  source: string,
  relativeFile: string,
  basePath: string,
  destinations: ReadonlyMap<string, LensDestination>
): Promise<string> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkFrontmatter, ['yaml'])
    .use(remarkGfm)
    .use(remarkLensLinks, {destinations})
    .use(remarkRehype, {allowDangerousHtml: true})
    .use(rehypeSlug)
    .use(rehypeRewriteLinks, {relativeFile, basePath})
    .use(rehypeShiki, {
      themes: {light: 'github-light', dark: 'github-dark'},
      defaultColor: false,
      fallbackLanguage: 'text',
      lazy: true
    })
    .use(rehypeStringify, {allowDangerousHtml: true})
    .process(source);
  return String(file);
}

type LensDestination = {href: string; title: string};

const websiteDirectory = fileURLToPath(new URL('../', import.meta.url));

async function loadLensDestinations(): Promise<ReadonlyMap<string, LensDestination>> {
  const destinations = new Map<string, LensDestination>();
  const modelsDirectory = path.join(websiteDirectory, 'src/lens/generated');
  for (const fileName of fs.readdirSync(modelsDirectory)) {
    if (!/^[\w.-]+\.json$/.test(fileName)) continue;
    const scenarioId = fileName.slice(0, -'.json'.length);
    const title = scenarioId.slice(scenarioId.indexOf('.') + 1).replace(/[.-]/g, ' ');
    destinations.set(scenarioId, {href: '/lens/' + encodeURIComponent(scenarioId), title});
  }
  const entriesDirectory = path.join(websiteDirectory, 'src/catalog/entries');
  for (const fileName of fs.readdirSync(entriesDirectory).filter(name => name.endsWith('.ts'))) {
    for (const entry of await loadEntries(path.join(entriesDirectory, fileName))) {
      const {id, title, lensScenario} = entry;
      if (typeof id === 'string' && typeof title === 'string' && typeof lensScenario === 'string') {
        destinations.set(lensScenario, {href: '/fn/' + id, title});
      }
    }
  }
  return destinations;
}

/** Pages for published lens recordings that have no authored function card. */
export async function listLensRecordingPagePaths(): Promise<string[]> {
  return [...(await loadLensDestinations()).values()]
    .map(destination => destination.href)
    .filter(href => href.startsWith('/lens/'))
    .sort();
}

type MarkdownNode = {
  type: string;
  children?: MarkdownNode[];
  position?: {start: {offset?: number}; end: {offset?: number}};
  value?: string;
  url?: string;
};

const LENS_DIRECTIVE_LINE = /^ {0,3}::(lens-catalog|lens)\{(.*)\}[ \t]*$/;

function remarkLensLinks({destinations}: {destinations: ReadonlyMap<string, LensDestination>}) {
  const paragraphParser = unified().use(remarkParse).use(remarkGfm);
  return (tree: MarkdownNode, file: {toString(): string}) => {
    const source = file.toString();
    const visit = (node: MarkdownNode): MarkdownNode[] => {
      const start = node.position?.start.offset;
      const end = node.position?.end.offset;
      if (node.type !== 'paragraph' || start === undefined || end === undefined) {
        if (node.children) node.children = node.children.flatMap(visit);
        return [node];
      }

      // Read the original lines so quoted captions and adjacent directives survive Markdown
      // inline parsing. Fenced and indented code never reach this paragraph branch.
      const lines = source.slice(start, end).split(/\r?\n/);
      if (!lines.some(line => LENS_DIRECTIVE_LINE.test(line))) return [node];

      const replacement: MarkdownNode[] = [];
      let prose: string[] = [];
      const flushProse = () => {
        if (!prose.length) return;
        replacement.push(...paragraphParser.parse(prose.join('\n')).children);
        prose = [];
      };
      for (const line of lines) {
        const directive = LENS_DIRECTIVE_LINE.exec(line);
        if (!directive) {
          prose.push(line);
          continue;
        }
        flushProse();
        const scenarioId = /(?:^|\s)scenario=([\w.-]+)(?=\s|$)/.exec(directive[2])?.[1];
        const rung = /(?:^|\s)rung=([\w-]+)(?=\s|$)/.exec(directive[2])?.[1];
        const entry = /(?:^|\s)entry=([\w-]+)(?=\s|$)/.exec(directive[2])?.[1];
        const destination = scenarioId ? destinations.get(scenarioId) : undefined;
        const content: MarkdownNode =
          directive[1] === 'lens-catalog'
            ? {
                type: 'link',
                url: '/explore/',
                children: [
                  {
                    type: 'text',
                    value: entry
                      ? 'Browse the ' + entry.replaceAll('-', ' ') + ' catalog'
                      : 'Browse the analysis catalog'
                  }
                ]
              }
            : destination
              ? {
                  type: 'link',
                  url: destination.href,
                  children: [
                    {
                      type: 'text',
                      value: 'View ' + destination.title + (rung ? ' (' + rung + ')' : '')
                    }
                  ]
                }
              : {type: 'text', value: 'Recording unavailable for ' + (scenarioId ?? 'this lens')};
        replacement.push({type: 'paragraph', children: [content]});
      }
      flushProse();
      return replacement;
    };
    visit(tree);
  };
}

function getTitle(source: string, relativeFile: string): string {
  const frontmatter = /^---\n([\s\S]*?)\n---/.exec(source);
  if (frontmatter) {
    const data = parseYaml(frontmatter[1]) as {title?: string} | null;
    if (data?.title) {
      return data.title;
    }
  }
  const heading = /^#\s+(.+)$/m.exec(source);
  if (heading) {
    return heading[1].replace(/`/g, '').trim();
  }
  return getSlugFromFile(relativeFile).split('/').pop() || 'Docs';
}

/** Rewrites markdown links into docs routes and applies the configured base to site routes. */
function rehypeRewriteLinks({relativeFile, basePath}: {relativeFile: string; basePath: string}) {
  const directory = path.posix.dirname(relativeFile);
  const routePrefix = basePath.replace(/\/$/, '');
  return (tree: HastNode) => {
    const visit = (node: HastNode) => {
      if (node.tagName === 'a' && typeof node.properties?.['href'] === 'string') {
        node.properties['href'] = rewriteHref(node.properties['href'], directory, routePrefix);
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}

function rewriteHref(href: string, directory: string, routePrefix: string): string {
  if (/^[a-z]+:/i.test(href) || href.startsWith('#') || href.startsWith('//')) {
    return href;
  }
  const [target, hash] = href.split('#');
  if (!target.endsWith('.md') && !target.startsWith('/docs/')) {
    if (
      target.startsWith('/') &&
      routePrefix &&
      target !== routePrefix &&
      !target.startsWith(`${routePrefix}/`)
    ) {
      return `${routePrefix}${href}`;
    }
    return href;
  }
  const docsPath = target.startsWith('/docs/')
    ? target.slice('/docs/'.length)
    : path.posix.normalize(path.posix.join(directory, target));
  const slug = getSlugFromFile(docsPath.endsWith('.md') ? docsPath : `${docsPath}.md`);
  return `${routePrefix}/docs/${slug}${hash ? `#${hash}` : ''}`;
}
