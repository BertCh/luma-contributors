// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Live WebGPU canvases are expensive, so card grids share a few "live slots". A card holds a slot
// while it is hovered, focused or near the middle of the viewport; the oldest holder is evicted.

import {useCallback, useEffect, useRef, useState} from 'react';

const MAX_LIVE_CANVASES = 3;

type SlotHolder = {id: number; release: () => void};

const holders: SlotHolder[] = [];
let nextHolderId = 1;

function acquireSlot(release: () => void): number {
  const id = nextHolderId++;
  holders.push({id, release});
  while (holders.length > MAX_LIVE_CANVASES) {
    holders.shift()?.release();
  }
  return id;
}

function releaseSlot(id: number): void {
  const index = holders.findIndex(holder => holder.id === id);
  if (index >= 0) {
    holders.splice(index, 1);
  }
}

export type LiveSlot = {
  /** True while this element owns a live canvas. */
  live: boolean;
  /** Attach to the element whose visibility requests a slot. */
  ref: (element: HTMLElement | null) => void;
  /** Spread on the element to request a slot on hover and focus. */
  handlers: {
    onPointerEnter: () => void;
    onPointerLeave: () => void;
    onFocus: () => void;
    onBlur: () => void;
  };
};

export function useLiveSlot(): LiveSlot {
  const [live, setLive] = useState(false);
  const slotId = useRef<number | null>(null);
  const element = useRef<HTMLElement | null>(null);
  const observer = useRef<IntersectionObserver | null>(null);

  const release = useCallback(() => {
    if (slotId.current !== null) {
      releaseSlot(slotId.current);
      slotId.current = null;
    }
    setLive(false);
  }, []);

  const acquire = useCallback(() => {
    if (slotId.current !== null) {
      return;
    }
    slotId.current = acquireSlot(() => {
      slotId.current = null;
      setLive(false);
    });
    setLive(true);
  }, []);

  const ref = useCallback(
    (node: HTMLElement | null) => {
      observer.current?.disconnect();
      element.current = node;
      if (node && typeof IntersectionObserver !== 'undefined') {
        // Only cards crossing the middle band of the viewport become live on their own.
        observer.current = new IntersectionObserver(
          entries => {
            for (const entry of entries) {
              if (entry.isIntersecting) {
                acquire();
              } else {
                release();
              }
            }
          },
          {rootMargin: '-35% 0px -35% 0px'}
        );
        observer.current.observe(node);
      }
    },
    [acquire, release]
  );

  useEffect(
    () => () => {
      observer.current?.disconnect();
      release();
    },
    [release]
  );

  return {
    live,
    ref,
    handlers: {onPointerEnter: acquire, onPointerLeave: release, onFocus: acquire, onBlur: release}
  };
}
