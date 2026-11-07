// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';
import {useEffect, useRef, useState} from 'react';
import type {CatalogEntryReference} from '../../catalog';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {getDemoParameters, getDemoScene} from '../../gpu/demos';
import {getPreviewParameters, getPreviewScene, PreviewFallback} from '../../gpu/previews';
import type {GpuSceneParameter} from '../../gpu/types';

export const DEFAULT_PREVIEW_SEED = 7;

/** A stable seed from the entry id, so entries of one preview kind still draw different pictures. */
export function getEntrySeed(entry: CatalogEntryReference): number {
  if (entry.preview.seed !== undefined) {
    return entry.preview.seed;
  }
  let hash = 2166136261;
  for (let index = 0; index < entry.id.length; index++) {
    hash = Math.imul(hash ^ entry.id.charCodeAt(index), 16777619);
  }
  return (hash >>> 0) % 997;
}

/** Tile-grid placeholder shown in prerendered HTML until the thumbnail mounts. */
function PreviewSkeleton() {
  return <div className="entry-preview-skeleton" aria-hidden="true" />;
}

/**
 * Card grids hold hundreds of thumbnails; inlining every SVG would make the prerendered page
 * megabytes. The thumbnail mounts on the client once its container nears the viewport.
 */
function DeferredFallback({children}: {children: ReactNode}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isNear, setIsNear] = useState(false);
  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof IntersectionObserver === 'undefined') {
      setIsNear(true);
      return undefined;
    }
    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(entry => entry.isIntersecting)) {
          setIsNear(true);
          observer.disconnect();
        }
      },
      {rootMargin: '400px'}
    );
    observer.observe(container);
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={containerRef} style={{position: 'absolute', inset: 0}}>
      {isNear ? children : <PreviewSkeleton />}
    </div>
  );
}

/** Slider parameters of an entry's picture: its demo's when it has a demo, else its preview kind's. */
export function getEntryParameters(entry: CatalogEntryReference): readonly GpuSceneParameter[] {
  if (getDemoScene(entry.id)) {
    return getDemoParameters(entry.symbol) ?? getDemoParameters(entry.id) ?? [];
  }
  return getPreviewParameters(entry.preview.kind) ?? [];
}

/** The picture of an entry: its flagship demo when one exists, else the generic scene for its kind. */
export function EntryPreview({
  entry,
  live = true,
  aspectRatio = 16 / 10,
  controls = false,
  interactive = false
}: {
  entry: CatalogEntryReference;
  /** When false only the static SVG is rendered (cards without a live slot). */
  live?: boolean;
  aspectRatio?: number;
  /** Render one slider per scene parameter (function pages). */
  controls?: boolean;
  /** Forward the pointer to the scene (hover animation on cards). */
  interactive?: boolean;
}) {
  const seed = getEntrySeed(entry);
  const fallback = <PreviewFallback kind={entry.preview.kind} seed={seed} />;
  if (!live) {
    return (
      <div className="entry-preview" style={{aspectRatio, position: 'relative'}}>
        <DeferredFallback>{fallback}</DeferredFallback>
      </div>
    );
  }
  return (
    <GpuCanvas
      scene={getDemoScene(entry.id) ?? getPreviewScene(entry.preview.kind)}
      fallback={fallback}
      seed={seed}
      label={`${entry.title}: ${entry.preview.caption ?? entry.summary}`}
      aspectRatio={aspectRatio}
      controls={controls}
      interactive={interactive}
      parameters={controls ? getEntryParameters(entry) : undefined}
    />
  );
}
