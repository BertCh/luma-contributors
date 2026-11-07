// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useSyncExternalStore} from 'react';
import {ResidentTooltip} from './resident-tooltip';
import {
  getInterpolationSurfaceTooltipId,
  type InterpolationSurfaceHoverStore
} from './interpolation-surface-hover';

const readNoOriginalValues = async () => ({});

/**
 * Popover for the hovered interpolation cell. It reuses ResidentTooltip, so the value comes from
 * the registered computed-value provider and shares the native tooltip styling and states.
 * Dismissible with Escape (WCAG 1.4.13); it never takes pointer events from the map.
 */
export function InterpolationSurfaceTooltip({store}: {store: InterpolationSurfaceHoverStore}) {
  const state = useSyncExternalStore(store.subscribe, store.getState, store.getState);
  useEffect(() => {
    if (!state) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') store.clear();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [store, Boolean(state)]);
  if (!state) return null;
  return (
    <div
      data-interpolation-surface-tooltip
      role="tooltip"
      style={{
        position: 'absolute',
        left: state.x + 12,
        top: state.y + 12,
        zIndex: 12,
        pointerEvents: 'none',
        background: 'var(--kepler-tooltip-background, #fff)',
        color: 'var(--kepler-tooltip-color, #111)',
        padding: '6px 8px',
        borderRadius: 4,
        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
        maxWidth: 280,
        fontSize: 12
      }}
    >
      <ResidentTooltip
        key={`${state.surfaceId}:${state.cellIndex}`}
        datasetId={getInterpolationSurfaceTooltipId(state.surfaceId)}
        layerId={`${state.surfaceId}-surface`}
        sourceRowIndex={state.cellIndex}
        readTooltip={readNoOriginalValues}
        fieldNames={[]}
        revisionKey={String(state.revision)}
      />
    </div>
  );
}
