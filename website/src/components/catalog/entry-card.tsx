// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import type {CatalogEntryReference} from '../../catalog';
import {Card} from '../ui/card';
import {DataFlow} from './data-flow';
import {EntryPreview} from './entry-preview';
import {useLiveSlot} from './use-live-slot';
import './catalog.css';

/** A visual card: static thumbnail that becomes a live canvas on hover, focus or when centered. */
export function EntryCard({entry}: {entry: CatalogEntryReference}) {
  const {live, ref, handlers} = useLiveSlot();
  return (
    <div className="entry-card" ref={ref} {...handlers}>
      <Card
        interactive
        media={
          <>
            <EntryPreview entry={entry} live={live} aspectRatio={16 / 10} />
            {entry.maturity !== 'stable' ? (
              <span className="entry-card-badge">{entry.maturity}</span>
            ) : null}
          </>
        }
        title={
          <Link to="/fn/$id" params={{id: entry.id}}>
            {entry.title}
          </Link>
        }
      >
        <p className="entry-card-summary">{entry.summary}</p>
        <DataFlow input={entry.dataTypes.input} output={entry.dataTypes.output} compact />
      </Card>
    </div>
  );
}
