// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ReactNode} from 'react';
import {useEffect, useRef, useState} from 'react';
import {getEntrySeed} from '../../catalog/entry-seed';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {getPreviewParameters, getPreviewScene, PreviewFallback} from '../../gpu/previews';
import type {EntryPreviewProps} from './entry-preview';

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

export default function IllustrativePreview({
  entry,
  live,
  aspectRatio,
  controls,
  interactive
}: EntryPreviewProps) {
  const seed = getEntrySeed(entry);
  const fallback = <PreviewFallback kind={entry.preview.kind} seed={seed} entryId={entry.id} />;
  if (!live) {
    return (
      <div className="entry-preview" style={{aspectRatio, position: 'relative'}}>
        <DeferredFallback>{fallback}</DeferredFallback>
      </div>
    );
  }
  return (
    <GpuCanvas
      scene={getPreviewScene(entry.preview.kind, entry.id)}
      fallback={fallback}
      seed={seed}
      label={`${entry.title}: ${entry.summary}`}
      aspectRatio={aspectRatio}
      controls={controls}
      interactive={interactive}
      pointerSurface={interactive && !controls ? 'card' : 'canvas'}
      parameters={controls ? getPreviewParameters(entry.preview.kind, entry.id) : undefined}
    />
  );
}
