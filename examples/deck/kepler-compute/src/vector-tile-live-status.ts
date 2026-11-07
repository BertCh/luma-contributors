// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Small status badge for vector-tile viewport populations; shows the analysis-scope refusal. */

import type {VectorTileLiveController} from './vector-tile-live-controller';

const BADGE_ID = 'kepler-vector-tile-scope';

export function renderVectorTileScopeStatus(controller: VectorTileLiveController): void {
  if (typeof document === 'undefined') return;
  const states = controller.getState();
  let badge = document.getElementById(BADGE_ID);
  if (states.length === 0) {
    badge?.remove();
    return;
  }
  if (!badge) {
    badge = document.createElement('div');
    badge.id = BADGE_ID;
    badge.setAttribute('role', 'status');
    badge.setAttribute('data-testid', 'vector-tile-scope');
    Object.assign(badge.style, {
      position: 'fixed',
      left: '8px',
      bottom: '8px',
      zIndex: '1000',
      maxWidth: '420px',
      padding: '6px 10px',
      font: '12px sans-serif',
      background: 'rgba(30, 36, 46, 0.92)',
      color: '#fff',
      borderRadius: '4px',
      pointerEvents: 'none'
    });
    document.body.append(badge);
  }
  badge.textContent = states
    .map(
      state =>
        `${state.scope?.label ?? state.layerId}: ${state.passingFeatureCount} of ${state.featureCount} features. ` +
        `Global analysis refused: ${state.completeDatasetDecision?.reason ?? 'viewport working set'}`
    )
    .join('\n');
  badge.style.whiteSpace = 'pre-line';
}
