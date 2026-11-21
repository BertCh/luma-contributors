// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dirname, join, resolve} from 'node:path';
import type {Element, Root, RootContent} from 'hast';
import {toString as getText} from 'hast-util-to-string';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkFrontmatter from 'remark-frontmatter';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import {unified} from 'unified';

// Match the site's Markdown-to-HAST and heading-slug pipeline, including rendered link labels.
const processor = unified()
  .use(remarkParse)
  .use(remarkFrontmatter, ['yaml'])
  .use(remarkGfm)
  .use(remarkRehype, {allowDangerousHtml: true})
  .use(rehypeSlug);

export function getRenderedWalkthroughDestinations(markdown: string) {
  const links: {label: string; href: string}[] = [];
  const anchors: string[] = [];
  const visit = (node: Root | RootContent) => {
    if (node.type === 'element') {
      const element: Element = node;
      if (element.properties['id']) anchors.push(String(element.properties['id']));
      if (element.tagName === 'a' && element.properties['href'])
        links.push({
          label: getText(element),
          href: String(element.properties['href'])
        });
    }
    if ('children' in node) for (const child of node.children) visit(child);
  };
  visit(processor.runSync(processor.parse(markdown)) as Root);
  return {links, anchors};
}

export function resolveWalkthroughTarget(root: string, page: string, href: string) {
  const [path, anchor] = href.split('#');
  return {path: resolve(dirname(join(root, page)), decodeURIComponent(path)), anchor};
}
