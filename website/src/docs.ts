// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Docs registry. Markdown is compiled to HTML at build time by build/markdown-plugin.ts; each page
// is its own lazily loaded chunk.

import docsIndex from 'virtual:docs-index';
import tableOfContents from '../../docs/table-of-contents.json';

export type DocPage = {slug: string; title: string; html: string};

export type SidebarItem =
  | {type: 'page'; slug: string; title: string}
  | {type: 'category'; label: string; items: SidebarItem[]};

type TableOfContentsItem = string | {type: string; label: string; items: TableOfContentsItem[]};

const pageLoaders = import.meta.glob<DocPage>('../../docs/**/*.md', {import: 'default'});

const loadersBySlug = new Map(
  Object.entries(pageLoaders).map(([file, loader]) => [getSlugFromGlobKey(file), loader])
);

const titlesBySlug = new Map(docsIndex.map(entry => [entry.slug, entry.title]));

export const DEFAULT_DOC_SLUG = 'api';

export async function loadDocPage(slug: string): Promise<DocPage | null> {
  const loader = loadersBySlug.get(slug.replace(/\/$/, '') || DEFAULT_DOC_SLUG);
  return loader ? loader() : null;
}

export function getSidebar(): SidebarItem[] {
  return (tableOfContents as TableOfContentsItem[]).map(convertTableOfContentsItem);
}

function convertTableOfContentsItem(item: TableOfContentsItem): SidebarItem {
  if (typeof item === 'string') {
    const slug = item.replace(/(^|\/)README$/, '');
    return {type: 'page', slug, title: titlesBySlug.get(slug) ?? slug};
  }
  return {type: 'category', label: item.label, items: item.items.map(convertTableOfContentsItem)};
}

function getSlugFromGlobKey(file: string): string {
  return file
    .replace(/^(\.\.\/)+docs\//, '')
    .replace(/\.md$/, '')
    .replace(/(^|\/)README$/, '');
}
