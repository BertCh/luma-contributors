// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The catalog index. Cards, filters and counts use lightweight summaries from
// `virtual:catalog-summaries` (build/catalog-plugin.ts). Full entries, with explanation, params,
// outputs and example, are loaded per group only by function pages (`loadEntry`).

import catalogSummaries from 'virtual:catalog-summaries';
import type {CatalogEntry} from './schema';
import {attachCatalogAnalysisMetadata, getCatalogDataTypes} from './analysis-metadata';
import {
  QUESTIONS,
  DATA_TYPES,
  DOMAINS,
  USE_CASES,
  type DataTypeId,
  type DomainId,
  type Facet,
  type UseCaseId
} from './taxonomy';

export type {CatalogEntry} from './schema';

/** Everything about an entry except the detail fields shown only on its function page. */
export type CatalogEntryReference = Omit<
  CatalogEntry,
  'explanation' | 'params' | 'moreRequired' | 'outputs' | 'example' | 'interpretation'
>;

export type CatalogSummary = CatalogEntryReference & {
  /** Entry file (src/catalog/entries/<group>.ts) that holds the full entry. */
  group: string;
};

const groupLoaders = import.meta.glob<CatalogEntry[]>('./entries/*.ts', {import: 'entries'});

/** Every entry summary, sorted by title. */
export const catalog: CatalogSummary[] = [...catalogSummaries].sort((left, right) =>
  left.title.localeCompare(right.title)
);

const entriesById = new Map(catalog.map(entry => [entry.id, entry]));

export function getEntry(id: string): CatalogSummary | undefined {
  return entriesById.get(id);
}

/** The full entry, loading its group chunk on demand. */
export async function loadEntry(id: string): Promise<CatalogEntry | undefined> {
  const summary = entriesById.get(id);
  const loadGroup = summary && groupLoaders[`./entries/${summary.group}.ts`];
  if (!loadGroup) {
    return undefined;
  }
  const entries = await loadGroup();
  const entry = entries.find(candidate => candidate.id === id);
  return entry && attachCatalogAnalysisMetadata(entry);
}

export type FacetKind = 'question' | 'data' | 'use' | 'domain';

export type CatalogFilters = {
  question?: string;
  dataType?: DataTypeId | string;
  useCase?: UseCaseId | string;
  domain?: DomainId | string;
  query?: string;
};

export const FACETS: Record<FacetKind, readonly Facet[]> = {
  question: QUESTIONS,
  data: DATA_TYPES,
  use: USE_CASES,
  domain: DOMAINS
};

export const FACET_TITLES: Record<FacetKind, string> = {
  question: 'Question',
  data: 'Data type',
  use: 'Use case',
  domain: 'Domain'
};

export function getFacet(kind: FacetKind, id: string): Facet | undefined {
  return FACETS[kind].find(facet => facet.id === id);
}

function matchesDataType(entry: CatalogEntryReference, dataType: string): boolean {
  const ports = getCatalogDataTypes(entry);
  return (
    ports.input.includes(dataType as DataTypeId) || ports.output.includes(dataType as DataTypeId)
  );
}

function matchesQuery(entry: CatalogEntryReference, query: string): boolean {
  const haystack = [entry.title, entry.symbol, entry.summary, entry.id].join(' ').toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every(word => haystack.includes(word));
}

/** Title matches first, then symbol matches, then the rest; stable within each rank. */
function rankByQuery<T extends CatalogEntryReference>(entries: T[], query: string): T[] {
  const normalizedQuery = query.trim().toLowerCase();
  const getRank = (entry: T): number => {
    if (entry.title.toLowerCase().includes(normalizedQuery)) return 0;
    if (entry.symbol.toLowerCase().includes(normalizedQuery)) return 1;
    return 2;
  };
  return entries
    .map((entry, index) => ({entry, index, rank: getRank(entry)}))
    .sort((left, right) => left.rank - right.rank || left.index - right.index)
    .map(({entry}) => entry);
}

/** Entries matching every given filter. A data type matches an entry's inputs or outputs. */
export function filterEntries(
  filters: CatalogFilters = {},
  source: CatalogSummary[] = catalog
): CatalogSummary[] {
  const {question, dataType, useCase, domain, query} = filters;
  const matches = source.filter(
    entry =>
      (!question || entry.questions?.some(identifier => identifier === question)) &&
      (!dataType || matchesDataType(entry, dataType)) &&
      (!useCase || entry.useCases.includes(useCase as UseCaseId)) &&
      (!domain || entry.domains.includes(domain as DomainId)) &&
      (!query || matchesQuery(entry, query))
  );
  return query ? rankByQuery(matches, query) : matches;
}

export function getEntriesBy(kind: FacetKind, id: string): CatalogSummary[] {
  return filterEntries(
    kind === 'question'
      ? {question: id}
      : kind === 'data'
        ? {dataType: id}
        : kind === 'use'
          ? {useCase: id}
          : {domain: id}
  );
}

export type FacetCounts = Record<FacetKind, Record<string, number>>;

/**
 * Counts per facet value under the other active filters (a facet ignores its own selection, so
 * the chips of one facet show what selecting each value would give).
 */
export function getFacetCounts(filters: CatalogFilters = {}): FacetCounts {
  const countFacet = (kind: FacetKind, without: keyof CatalogFilters): Record<string, number> => {
    const matching = filterEntries({...filters, [without]: undefined});
    const counts: Record<string, number> = {};
    for (const facet of FACETS[kind]) {
      counts[facet.id] = 0;
    }
    for (const entry of matching) {
      const ids =
        kind === 'question'
          ? (entry.questions ?? [])
          : kind === 'data'
            ? new Set<string>(Object.values(getCatalogDataTypes(entry)).flat())
            : kind === 'use'
              ? entry.useCases
              : entry.domains;
      for (const id of ids) {
        counts[id] = (counts[id] ?? 0) + 1;
      }
    }
    return counts;
  };
  return {
    question: countFacet('question', 'question'),
    data: countFacet('data', 'dataType'),
    use: countFacet('use', 'useCase'),
    domain: countFacet('domain', 'domain')
  };
}

export function resolveRelated(entry: CatalogEntryReference): CatalogSummary[] {
  return (entry.related ?? [])
    .map(getEntry)
    .filter((related): related is CatalogSummary => Boolean(related));
}

// "Table" matches nearly everything, so shared tables count for little.
const GENERIC_DATA_TYPE_WEIGHT = 0.25;

function scoreSharedTypes(
  produced: readonly DataTypeId[],
  consumed: readonly DataTypeId[]
): number {
  let score = 0;
  for (const dataType of produced) {
    if (consumed.includes(dataType)) {
      score += dataType === 'table' ? GENERIC_DATA_TYPE_WEIGHT : 1;
    }
  }
  return score;
}

/** Chaining only makes sense inside a field: neighbors must share a domain. */
function sharesDomain(entry: CatalogEntryReference, other: CatalogEntryReference): boolean {
  return other.domains.some(domain => entry.domains.includes(domain));
}

/** What to do next: entries whose inputs match this entry's outputs, best match first. */
export function getNextSteps(entry: CatalogEntryReference, limit = 6): CatalogSummary[] {
  return catalog
    .filter(other => other.id !== entry.id && sharesDomain(entry, other))
    .map(other => ({
      other,
      score: scoreSharedTypes(getCatalogDataTypes(entry).output, getCatalogDataTypes(other).input)
    }))
    .filter(({score}) => score > 0)
    .sort(
      (left, right) => right.score - left.score || left.other.title.localeCompare(right.other.title)
    )
    .slice(0, limit)
    .map(({other}) => other);
}

/** What feeds this entry: entries whose outputs match this entry's inputs. */
export function getPreviousSteps(entry: CatalogEntryReference, limit = 6): CatalogSummary[] {
  return catalog
    .filter(other => other.id !== entry.id && sharesDomain(entry, other))
    .map(other => ({
      other,
      score: scoreSharedTypes(getCatalogDataTypes(other).output, getCatalogDataTypes(entry).input)
    }))
    .filter(({score}) => score > 0)
    .sort(
      (left, right) => right.score - left.score || left.other.title.localeCompare(right.other.title)
    )
    .slice(0, limit)
    .map(({other}) => other);
}
