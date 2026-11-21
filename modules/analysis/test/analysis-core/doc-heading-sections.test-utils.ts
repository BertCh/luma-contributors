// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Element, Root, RootContent} from 'hast';
import {toString as getHeadingText} from 'hast-util-to-string';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkFrontmatter from 'remark-frontmatter';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import {unified} from 'unified';

// Same parsing, heading text and slug plugins as website/build/markdown-plugin.ts.
const DOC_HEADING_PROCESSOR = unified()
  .use(remarkParse)
  .use(remarkFrontmatter, ['yaml'])
  .use(remarkGfm)
  .use(remarkRehype, {allowDangerousHtml: true})
  .use(rehypeSlug);
const docSectionsByMarkdown = new Map<string, {heading: string; anchor: string; body: string}[]>();

/** Keep renderer anchors while excluding whole generated blocks from contributor evidence. */
export function getDocSections(
  markdown: string
): {heading: string; anchor: string; body: string}[] {
  const cached = docSectionsByMarkdown.get(markdown);
  if (cached) return cached;
  const generated = [
    ...markdown.matchAll(/\{\/\*\s*parameters:[\s\S]*?(?:\{\/\*\s*\/parameters\s*\*\/\}|$)/g)
  ];
  const cleanEvidence = (text: string) =>
    text
      .replace(/\{\/\*\s*parameters:[\s\S]*?(?:\{\/\*\s*\/parameters\s*\*\/\}|$)/g, '')
      .replace(/^(```|~~~)[\s\S]*?^\1[^\n]*$/gm, '')
      .replace(/\{?\/\*[\s\S]*?\*\/\}?|<!--[\s\S]*?-->/g, '');
  const tree = DOC_HEADING_PROCESSOR.runSync(DOC_HEADING_PROCESSOR.parse(markdown)) as Root;
  const headings: Element[] = [];
  const visit = (node: Root | RootContent) => {
    if (node.type === 'element' && /^h[1-6]$/.test(node.tagName)) headings.push(node);
    if ('children' in node) for (const child of node.children) visit(child);
  };
  visit(tree);
  const excludedRanges = [
    ...generated.map(block => ({start: block.index!, end: block.index! + block[0].length})),
    ...headings.flatMap((heading, index) => {
      if (!/ parameters$/i.test(getHeadingText(heading))) return [];
      const next = headings
        .slice(index + 1)
        .find(candidate => candidate.tagName <= heading.tagName);
      return [
        {
          start: heading.position?.start.offset ?? 0,
          end: next?.position?.start.offset ?? markdown.length
        }
      ];
    })
  ].sort((first, second) => first.start - second.start);
  // Heading titles identify sections; only prose/tables can explain the contributor. Keep all
  // rendered heading ranges out of evidence while preserving their real anchors above.
  const evidenceRanges = [
    ...excludedRanges,
    ...headings.map(heading => ({
      start: heading.position?.start.offset ?? 0,
      end: heading.position?.end.offset ?? 0
    }))
  ].sort((first, second) => first.start - second.start);
  const getEvidence = (start: number, end: number): string => {
    let cursor = start;
    const parts: string[] = [];
    for (const range of evidenceRanges) {
      if (range.end <= cursor || range.start >= end) continue;
      parts.push(markdown.slice(cursor, Math.max(cursor, range.start)));
      cursor = Math.min(end, Math.max(cursor, range.end));
    }
    parts.push(markdown.slice(cursor, end));
    return cleanEvidence(parts.join('\n'));
  };
  const sections = headings.flatMap((heading, index) => {
    const start = heading.position?.start.offset;
    const end = heading.position?.end.offset;
    if (
      start === undefined ||
      end === undefined ||
      excludedRanges.some(range => start >= range.start && start < range.end)
    )
      return [];
    const next = headings.slice(index + 1).find(candidate => candidate.tagName <= heading.tagName);
    return [
      {
        heading: getHeadingText(heading),
        anchor: String(heading.properties['id']),
        body: getEvidence(end, next?.position?.start.offset ?? markdown.length)
      }
    ];
  });
  docSectionsByMarkdown.set(markdown, sections);
  return sections;
}

const renderedDestinations = new Map<
  string,
  {anchors: string[]; links: {href: string; line: number}[]}
>();

/** Rendered links and all rendered anchors, including generated headings. */
export function getRenderedDocDestinations(markdown: string) {
  const cached = renderedDestinations.get(markdown);
  if (cached) return cached;
  const tree = DOC_HEADING_PROCESSOR.runSync(DOC_HEADING_PROCESSOR.parse(markdown)) as Root;
  const anchors: string[] = [];
  const links: {href: string; line: number}[] = [];
  const visit = (node: Root | RootContent) => {
    if (node.type === 'element') {
      if (node.properties['id']) anchors.push(String(node.properties['id']));
      if (node.tagName === 'a' && node.properties['href'])
        links.push({href: String(node.properties['href']), line: node.position?.start.line ?? 1});
    }
    if ('children' in node) for (const child of node.children) visit(child);
  };
  visit(tree);
  const result = {anchors, links};
  renderedDestinations.set(markdown, result);
  return result;
}

/** Tables after the same Markdown/GFM parsing used by the site, with section and column roles. */
export function getRenderedDocTableRows(markdown: string) {
  const tree = DOC_HEADING_PROCESSOR.runSync(DOC_HEADING_PROCESSOR.parse(markdown)) as Root;
  const sections: string[] = [];
  const rows: {sections: string[]; headers: string[]; cells: string[]; text: string}[] = [];
  for (const node of tree.children) {
    if (node.type !== 'element') continue;
    if (/^h[1-6]$/.test(node.tagName)) {
      const level = Number(node.tagName.slice(1));
      sections.length = level - 1;
      sections[level - 1] = getHeadingText(node);
    }
    if (node.tagName !== 'table') continue;
    const tableRows: Element[] = [];
    const visit = (element: Element) => {
      if (element.tagName === 'tr') tableRows.push(element);
      for (const child of element.children) if (child.type === 'element') visit(child);
    };
    visit(node);
    const values = (row: Element) =>
      row.children
        .filter(
          (child): child is Element => child.type === 'element' && /^(td|th)$/.test(child.tagName)
        )
        .map(element => getHeadingText(element));
    const headers = values(tableRows[0]);
    for (const row of tableRows.slice(1)) {
      rows.push({
        sections: sections.filter(Boolean),
        headers,
        cells: values(row),
        text: markdown.slice(row.position?.start.offset, row.position?.end.offset)
      });
    }
  }
  return rows;
}
