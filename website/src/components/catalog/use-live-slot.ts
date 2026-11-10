// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState, type FocusEvent, type PointerEvent} from 'react';
import {rememberCardHover} from '../../gpu/canvas-pointer';

/** A card runs while deliberately hovered or keyboard-focused. A touch tap follows the link. */
export function useLiveSlot() {
  const ref = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(
    () => () => {
      clearTimeout(hoverTimer.current);
      clearTimeout(leaveTimer.current);
    },
    []
  );

  return {
    live: hovered || focused,
    ref,
    handlers: {
      onPointerEnter(event: PointerEvent<HTMLElement>) {
        if (event.pointerType === 'touch') return;
        rememberCardHover(event.currentTarget, [event.clientX, event.clientY]);
        clearTimeout(hoverTimer.current);
        clearTimeout(leaveTimer.current);
        // Avoid compiling graphs for every card the pointer crosses on its way to a destination.
        hoverTimer.current = setTimeout(() => setHovered(true), 90);
      },
      onPointerMove(event: PointerEvent<HTMLElement>) {
        if (event.pointerType !== 'touch') {
          rememberCardHover(event.currentTarget, [event.clientX, event.clientY]);
        }
      },
      onPointerLeave(event: PointerEvent<HTMLElement>) {
        rememberCardHover(event.currentTarget, null);
        clearTimeout(hoverTimer.current);
        // A short grace keeps the scene when the pointer crosses a gap or comes straight back.
        clearTimeout(leaveTimer.current);
        leaveTimer.current = setTimeout(() => setHovered(false), 200);
      },
      onFocus(event: FocusEvent<HTMLElement>) {
        setFocused(event.target.matches(':focus-visible'));
      },
      onBlur(event: FocusEvent<HTMLElement>) {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocused(false);
      }
    }
  };
}
