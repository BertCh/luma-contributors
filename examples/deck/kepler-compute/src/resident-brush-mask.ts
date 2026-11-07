// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toRadians} from '@math.gl/core';
import {MEAN_EARTH_RADIUS_METERS} from './earth-constants';

/**
 * Brushing is independent of the filter mask: it never changes which rows exist, only which are
 * highlighted. State carries its own revision so consumers can detect a stale brushed set.
 */
export type ResidentBrushState = {
  datasetId: string;
  revision: number;
  enabled: boolean;
  /** Brush center in geographic coordinates and radius in meters. */
  center: readonly [number, number] | null;
  radiusMeters: number;
};

export class ResidentBrushMask {
  private state = new Map<string, ResidentBrushState>();
  private listeners = new Set<() => void>();

  getState(datasetId: string): ResidentBrushState {
    return (
      this.state.get(datasetId) ?? {
        datasetId,
        revision: 0,
        enabled: false,
        center: null,
        radiusMeters: 0
      }
    );
  }

  setBrush(
    datasetId: string,
    center: readonly [number, number] | null,
    radiusMeters: number
  ): ResidentBrushState {
    const previous = this.getState(datasetId);
    const next: ResidentBrushState = {
      datasetId,
      revision: previous.revision + 1,
      enabled: center !== null && radiusMeters > 0,
      center,
      radiusMeters
    };
    this.state.set(datasetId, next);
    for (const listener of this.listeners) listener();
    return next;
  }

  clearBrush(datasetId: string): ResidentBrushState {
    return this.setBrush(datasetId, null, 0);
  }

  isCurrent(state: ResidentBrushState): boolean {
    return this.getState(state.datasetId).revision === state.revision;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

/** Haversine membership test for bounded CPU checks of an already-read row (not a scan). */
export function isPointBrushed(
  state: ResidentBrushState,
  longitude: number,
  latitude: number
): boolean {
  if (!state.enabled || !state.center) return false;
  const deltaLatitude = toRadians(latitude - state.center[1]);
  const deltaLongitude = toRadians(longitude - state.center[0]);
  const a =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(toRadians(state.center[1])) *
      Math.cos(toRadians(latitude)) *
      Math.sin(deltaLongitude / 2) ** 2;
  return 2 * MEAN_EARTH_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(a))) <= state.radiusMeters;
}
