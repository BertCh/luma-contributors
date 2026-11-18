// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compiles docs/**/*.md to HTML at build time. Each markdown import becomes a module exporting
// {slug, title, html}, so the browser never parses markdown. `virtual:docs-index` lists every page.

import fs from 'node:fs';
import path from 'node:path';
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
      const html = await renderMarkdown(source, relativeFile, basePath);
      const page = {slug, title: getTitle(source, relativeFile), html};
      return {code: `export default ${JSON.stringify(page)};`, map: null};
    }
  };
}

async function renderMarkdown(
  source: string,
  relativeFile: string,
  basePath: string
): Promise<string> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkFrontmatter, ['yaml'])
    .use(remarkGfm)
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
