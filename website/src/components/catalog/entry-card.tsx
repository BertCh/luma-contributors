// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import type {CatalogEntryReference} from '../../catalog';
import {getDemoData} from '../../gpu/data/provenance';
import {Card} from '../ui/card';
import {DataFlow} from './data-flow';
import {EntryPreview} from './entry-preview';
import {useLiveSlot} from './use-live-slot';
import './catalog.css';

/** A visual card: static thumbnail that becomes a live canvas on hover or keyboard focus. */
export function EntryCard({entry}: {entry: CatalogEntryReference}) {
  const {live, ref, handlers} = useLiveSlot();
  const {sources} = getDemoData(entry.id);
  // When a coastline accompanies observations, name the observations on the card.
  const observations = sources.filter(source => source.id !== 'natural-earth');
  const labels = (observations.length ? observations : sources).map(source => source.label);
  const dataset = labels.length ? labels.join(' · ') : 'Modeled scenario';
  return (
    <div className="entry-card" ref={ref} {...handlers}>
      <Card
        interactive
        meta={
          <span className="entry-card-dataset" title={dataset}>
            {dataset}
          </span>
        }
        media={
          <>
            <EntryPreview entry={entry} live={live} aspectRatio={16 / 10} interactive />
            {entry.maturity !== 'stable' ? (
              <span className="entry-card-badge">{entry.maturity}</span>
            ) : null}
          </>
        }
        title={
          <Link
            to="/fn/$id"
            params={{id: entry.id}}
            aria-description="Focus for a live preview. Use the arrow keys to explore it; Home returns it to its start."
            aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home"
          >
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
