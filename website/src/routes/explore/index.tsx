// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {createFileRoute} from '@tanstack/react-router';
import {catalog, filterEntries, getFacetCounts, type FacetKind} from '../../catalog';
import {EntryGrid} from '../../components/catalog/entry-grid';
import {FacetPicker} from '../../components/catalog/facet-picker';
import '../../components/catalog/catalog.css';
import {SITE_NAME} from '../__root';

const PAGE_SIZE = 36;

export type ExploreSearch = {q?: string; data?: string; use?: string; domain?: string};

function readText(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

export const Route = createFileRoute('/explore/')({
  validateSearch: (search: Record<string, unknown>): ExploreSearch => ({
    q: readText(search.q),
    data: readText(search.data),
    use: readText(search.use),
    domain: readText(search.domain)
  }),
  head: () => ({meta: [{title: `Explore | ${SITE_NAME}`}]}),
  component: ExplorePage
});

function ExplorePage() {
  // The prerendered HTML has no query string, so the first client render must not use it either.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);
  const urlSearch = Route.useSearch();
  const search: ExploreSearch = isMounted ? urlSearch : {};
  const navigate = Route.useNavigate();
  const filters = {
    dataType: search.data,
    useCase: search.use,
    domain: search.domain,
    query: search.q
  };
  const allEntries = filterEntries(filters);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const filterKey = [search.q, search.data, search.use, search.domain].join('|');
  // A new filter starts again from the first page.
  useEffect(() => setVisibleCount(PAGE_SIZE), [filterKey]);
  const entries = allEntries.slice(0, visibleCount);
  const counts = getFacetCounts(filters);
  const isFiltered = Boolean(search.q || search.data || search.use || search.domain);

  const updateSearch = (patch: Partial<ExploreSearch>) =>
    navigate({search: previous => ({...previous, ...patch}), replace: true, resetScroll: false});

  return (
    <main className="page explore-page">
      <header className="explore-header">
        <p className="eyebrow">Explore</p>
        <h1>{catalog.length} GPU operations</h1>
        <input
          type="search"
          className="explore-search"
          placeholder="Search: moran, buffer, flow, h3 ..."
          aria-label="Search functions"
          value={search.q ?? ''}
          onChange={event => updateSearch({q: event.target.value || undefined})}
        />
      </header>
      <FacetPicker
        selection={{data: search.data, use: search.use, domain: search.domain}}
        counts={counts}
        onSelect={(kind: FacetKind, id) =>
          updateSearch(kind === 'data' ? {data: id} : kind === 'use' ? {use: id} : {domain: id})
        }
      />
      <p className="explore-count" aria-live="polite">
        {allEntries.length} {allEntries.length === 1 ? 'function' : 'functions'}
        {isFiltered ? (
          <>
            {' · '}
            <button
              type="button"
              className="link-button"
              onClick={() => navigate({search: {}, replace: true})}
            >
              Clear filters
            </button>
          </>
        ) : null}
      </p>
      <EntryGrid entries={entries} emptyMessage="No function matches. Try fewer filters." />
      {allEntries.length > entries.length ? (
        <div className="explore-more">
          <button
            type="button"
            className="explore-more-button"
            onClick={() => setVisibleCount(count => count + PAGE_SIZE)}
          >
            Show {Math.min(PAGE_SIZE, allEntries.length - entries.length)} more
          </button>
          <span className="explore-count">
            {entries.length} of {allEntries.length}
          </span>
        </div>
      ) : null}
    </main>
  );
}
