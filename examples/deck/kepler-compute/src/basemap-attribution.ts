// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Visible attribution for the active basemap style.
 *
 * The attribution strings come from the `attribution` field of the style's sources (the same
 * data `basemap-context.ts` preserves). Text is reduced to plain text: tags are stripped and the
 * common HTML entities decoded, so a hostile style cannot inject markup into the host page.
 */

type StyleLike = {sources?: Record<string, {attribution?: string} | undefined>} | undefined;

const ENTITIES: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&copy;': '©',
  '&nbsp;': ' '
};

/** Converts an attribution HTML snippet to plain text. */
export function makeAttributionPlainText(html: string): string {
  return html
    .replace(/<[^>]*>/g, '')
    .replace(/&(amp|lt|gt|quot|#39|copy|nbsp);/g, entity => ENTITIES[entity] ?? entity)
    .replace(/\s+/g, ' ')
    .trim();
}

/** Distinct plain-text attributions of a style's sources, in source order. */
export function getStyleAttributionTexts(style: StyleLike): string[] {
  const texts: string[] = [];
  for (const source of Object.values(style?.sources ?? {})) {
    const text = source?.attribution ? makeAttributionPlainText(source.attribution) : '';
    if (text && !texts.includes(text)) texts.push(text);
  }
  return texts;
}

export type AttributionSegment = Readonly<{text: string; href?: string}>;

/**
 * Splits an attribution HTML snippet into text and anchor segments. Only http(s) hrefs are kept;
 * every other tag is dropped, so nothing from the style is ever parsed as markup by the page.
 */
export function parseAttributionSegments(html: string): AttributionSegment[] {
  const segments: AttributionSegment[] = [];
  const anchorPattern = /<a\b[^>]*?\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')[^>]*>([\s\S]*?)<\/a\s*>/gi;
  let cursor = 0;
  const pushText = (raw: string) => {
    const text = makeAttributionPlainText(raw);
    if (text) segments.push({text});
  };
  for (const match of html.matchAll(anchorPattern)) {
    pushText(html.slice(cursor, match.index));
    const href = (match[1] ?? match[2] ?? '').trim();
    const text = makeAttributionPlainText(match[3]);
    if (text) segments.push(/^https?:\/\//i.test(href) ? {text, href} : {text});
    cursor = (match.index ?? 0) + match[0].length;
  }
  pushText(html.slice(cursor));
  return segments;
}

/** Distinct attribution segment lists of a style's sources, keeping safe links. */
export function getStyleAttributionSegments(style: StyleLike): AttributionSegment[][] {
  const lists: AttributionSegment[][] = [];
  const seen = new Set<string>();
  for (const source of Object.values(style?.sources ?? {})) {
    if (!source?.attribution) continue;
    const segments = parseAttributionSegments(source.attribution);
    const key = segments.map(segment => segment.text).join(' ');
    if (key && !seen.has(key)) {
      seen.add(key);
      lists.push(segments);
    }
  }
  return lists;
}

type MapStyleStateLike = {
  styleType?: string;
  mapStyles?: Record<string, {style?: StyleLike} | undefined>;
};

/** Attribution texts of the style Kepler currently shows. */
export function getActiveBasemapAttributions(mapStyle: MapStyleStateLike | undefined): string[] {
  const styleType = mapStyle?.styleType;
  return styleType ? getStyleAttributionTexts(mapStyle?.mapStyles?.[styleType]?.style) : [];
}

/** Attribution segments (with safe links) of the style Kepler currently shows. */
export function getActiveBasemapAttributionSegments(
  mapStyle: MapStyleStateLike | undefined
): AttributionSegment[][] {
  const styleType = mapStyle?.styleType;
  return styleType ? getStyleAttributionSegments(mapStyle?.mapStyles?.[styleType]?.style) : [];
}

export const BASEMAP_ATTRIBUTION_ELEMENT_ID = 'basemap-attribution';

/** Shows (or hides, when empty) the attribution overlay. Uses textContent only. */
export function renderBasemapAttribution(
  attributions: readonly (string | readonly AttributionSegment[])[]
): void {
  if (typeof document === 'undefined') return;
  let element = document.getElementById(BASEMAP_ATTRIBUTION_ELEMENT_ID);
  if (!attributions.length) {
    element?.remove();
    return;
  }
  if (!element) {
    element = document.createElement('div');
    element.id = BASEMAP_ATTRIBUTION_ELEMENT_ID;
    element.setAttribute('role', 'note');
    Object.assign(element.style, {
      position: 'fixed',
      left: '8px',
      bottom: '4px',
      zIndex: '40',
      maxWidth: '50vw',
      padding: '1px 6px',
      font: '11px/1.4 sans-serif',
      color: '#222',
      background: 'rgba(255,255,255,0.8)',
      borderRadius: '3px',
      pointerEvents: 'auto'
    });
    document.body.appendChild(element);
  }
  element.replaceChildren('Basemap: ');
  attributions.forEach((attribution, index) => {
    if (index > 0) element.append(' | ');
    for (const segment of typeof attribution === 'string' ? [{text: attribution}] : attribution) {
      if (!('href' in segment) || !segment.href) {
        element.append(segment.text);
        continue;
      }
      const anchor = document.createElement('a');
      anchor.textContent = segment.text;
      anchor.href = segment.href;
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
      element.append(anchor);
    }
  });
}
