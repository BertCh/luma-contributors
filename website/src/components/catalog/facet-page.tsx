// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {FACET_TITLES, getEntriesBy, getFacet, type FacetKind} from '../../catalog';
import {FacetIcon} from '../brand/facet-icon';
import {EntryGrid} from './entry-grid';
import './catalog.css';

/** Landing page for one facet value: big icon, description, the grid. */
export function FacetPage({kind, id}: {kind: FacetKind; id: string}) {
  const facet = getFacet(kind, id);
  const entries = getEntriesBy(kind, id);
  const search = kind === 'data' ? {data: id} : kind === 'use' ? {use: id} : {domain: id};
  return (
    <main className="page facet-page">
      <header className="facet-hero">
        <div className="facet-hero-icon">
          <FacetIcon id={id} size={96} />
        </div>
        <div>
          <p className="eyebrow">
            <Link to="/explore">Explore</Link> / {FACET_TITLES[kind]}
          </p>
          <h1>{facet?.label ?? id}</h1>
          <p className="lead">{facet?.description}</p>
          <p className="facet-hero-meta">
            {entries.length} {entries.length === 1 ? 'function' : 'functions'} ·{' '}
            <Link to="/explore" search={search}>
              Combine with other filters
            </Link>
          </p>
        </div>
      </header>
      <EntryGrid entries={entries} emptyMessage="No functions are cataloged here yet." />
    </main>
  );
}
