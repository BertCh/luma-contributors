// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRenderedDocTableRows} from './doc-heading-sections.test-utils';

type RoadmapRow = ReturnType<typeof getRenderedDocTableRows>[number];

export function getIdentityTokens(text: string): string[] {
  const expanded = text.replace(
    /\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*)-(\d+)\s+to\s+\1-(\d+)\b/g,
    (_match, prefix: string, first: string, last: string) => {
      const start = Number(first);
      const end = Number(last);
      if (end < start || end - start > 1000) throw new Error('invalid roadmap ID range');
      return Array.from(
        {length: end - start + 1},
        (_, index) => `${prefix}-${String(start + index).padStart(first.length, '0')}`
      ).join(', ');
    }
  );
  // Components are opaque, case-sensitive IDs: X04 and raster-fields-05 are valid too.
  // Split only the compound separators, preserving every component in exact-set matching.
  return expanded
    .split(/\s*[,/]\s*/)
    .map(component => component.trim())
    .filter(Boolean);
}

export function getRoadmapRole(row: RoadmapRow): string {
  if (row.sections.includes('High-priority summary')) return 'summary';
  const priority = row.cells[row.headers.indexOf('Priority')];
  if (priority === 'Decision' || row.sections.includes('Decisions for the maintainer'))
    return 'decision';
  if (row.sections.includes('Known limits recorded by finished passes')) return 'limit';
  return 'implementation';
}

export function findRoadmapRows(markdown: string, identity: string, section: string, role: string) {
  const wanted = getIdentityTokens(identity);
  return getRenderedDocTableRows(markdown).filter(
    row =>
      row.headers[0] === (role === 'limit' ? 'Area' : 'ID') &&
      row.sections.includes(section) &&
      getRoadmapRole(row) === role &&
      new Set(wanted).size === new Set(getIdentityTokens(row.cells[0])).size &&
      wanted.every(token => getIdentityTokens(row.cells[0]).includes(token))
  );
}
