// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogEntryReference} from '../../catalog';
import {EntryCard} from './entry-card';
import './catalog.css';

export function EntryGrid({
  entries,
  emptyMessage = 'Nothing here yet.'
}: {
  entries: CatalogEntryReference[];
  emptyMessage?: string;
}) {
  if (entries.length === 0) {
    return <p className="entry-grid-empty">{emptyMessage}</p>;
  }
  return (
    <div className="entry-grid">
      {entries.map(entry => (
        <EntryCard key={entry.id} entry={entry} />
      ))}
    </div>
  );
}
