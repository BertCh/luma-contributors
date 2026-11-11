// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {lazy, Suspense} from 'react';
import type {CatalogEntryReference} from '../../catalog';
import {getEntrySeed} from '../../catalog/entry-seed';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {getDemo} from '../../gpu/demos';
import {getRunParameters, getRunScene, hasRun} from '../../gpu/runs/registry';
import {getPublicUrl} from '../../public-url';

// Only entries awaiting a real run need the illustration builders.
const IllustrativePreview = lazy(() => import('./illustrative-preview'));

export function hasOwnPicture(entry: CatalogEntryReference): boolean {
  return Boolean(getDemo(entry.id) || hasRun(entry.id));
}

/**
 * Static picture of a run, rendered at build time from the real operation (scripts/snapshot-runs.mjs).
 * Both themes are in the HTML; CSS shows the one that matches the page theme.
 */
function RunSnapshot({entry}: {entry: CatalogEntryReference}) {
  return (
    <div className="run-snapshot">
      {(['dark', 'light'] as const).map(theme => (
        <img
          key={theme}
          className={`run-snapshot-${theme}`}
          src={getPublicUrl(`runs/${entry.id}-${theme}.webp`)}
          width={640}
          height={360}
          alt=""
          loading="lazy"
          decoding="async"
        />
      ))}
    </div>
  );
}

export type EntryPreviewProps = {
  entry: CatalogEntryReference;
  live?: boolean;
  aspectRatio?: number;
  controls?: boolean;
  interactive?: boolean;
};

export function EntryPreview({
  entry,
  live = true,
  aspectRatio = 16 / 10,
  controls = false,
  interactive = false
}: EntryPreviewProps) {
  const demo = getDemo(entry.id);
  const scene = demo?.loader ?? getRunScene(entry.id);
  if (!scene) {
    return (
      <Suspense fallback={<div className="entry-preview" style={{aspectRatio}} />}>
        <IllustrativePreview {...{entry, live, aspectRatio, controls, interactive}} />
      </Suspense>
    );
  }
  const fallback = <RunSnapshot entry={entry} />;
  if (!live) {
    return (
      <div className="entry-preview" style={{aspectRatio, position: 'relative'}}>
        {fallback}
      </div>
    );
  }
  return (
    <GpuCanvas
      scene={scene}
      fallback={fallback}
      seed={getEntrySeed(entry)}
      label={`${entry.title}: ${entry.summary}`}
      aspectRatio={aspectRatio}
      controls={controls}
      interactive={interactive}
      pointerSurface={interactive && !controls ? 'card' : 'canvas'}
      parameters={controls ? (demo?.parameters ?? getRunParameters(entry.id)) : undefined}
    />
  );
}
