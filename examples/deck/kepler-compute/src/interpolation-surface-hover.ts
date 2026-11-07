// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeInterpolationTooltipValues} from './interpolation-tooltip';
import {
  registerTooltipComputedProvider,
  type ResidentTooltipComputedProvider,
  type ResidentTooltipValue
} from './resident-tooltip-fields';

/** One hovered cell of an interpolation surface, in screen pixels relative to the Deck canvas. */
export type InterpolationSurfaceHoverState = {
  surfaceId: string;
  cellIndex: number;
  x: number;
  y: number;
  revision: number;
};

export type InterpolationSurfacePick = {
  surfaceId: string;
  cellIndex: number;
  x: number;
  y: number;
};

type SurfaceCell = {
  cellIndex: number;
  longitude: number;
  latitude: number;
  value: number | null;
  neighbors?: number;
  variance?: number | null;
  lower?: number | null;
  upper?: number | null;
  confidenceLevel?: number;
};

type SurfaceSource = {
  /** Reads one grid cell; rejects when the surface is stale. */
  readInterpolationPage(offset: number, limit?: number): Promise<readonly SurfaceCell[]>;
  settings: {mode: string; column: string};
};

/** Synthetic tooltip dataset id; cell indexes are not source row indexes. */
export function getInterpolationSurfaceTooltipId(surfaceId: string): string {
  return `${surfaceId}::interpolation-surface`;
}

/**
 * Computed-value provider for hovering the surface itself: the "row" is the grid cell index.
 * One bounded cell read (value plus variance/interval for kriging, contributing neighbor count
 * for IDW); the cell center is reported so the value is tied to a location.
 */
export function makeInterpolationSurfaceProvider(
  source: SurfaceSource
): ResidentTooltipComputedProvider {
  return async (_datasetId, _layerId, cellIndex) => {
    const [cell] = await source.readInterpolationPage(cellIndex, 1).catch(() => []);
    if (!cell) return null;
    const values: Record<string, ResidentTooltipValue> = makeInterpolationTooltipValues(
      cell,
      source.settings.column
    );
    values['Cell center longitude'] = cell.longitude;
    values['Cell center latitude'] = cell.latitude;
    return values;
  };
}

/** Registers the surface provider; returns the unregister function. */
export function registerInterpolationSurfaceProvider(
  surfaceId: string,
  source: SurfaceSource
): () => void {
  return registerTooltipComputedProvider(
    getInterpolationSurfaceTooltipId(surfaceId),
    makeInterpolationSurfaceProvider(source)
  );
}

/** Single shared hover state: only one surface tooltip is visible at a time. */
export class InterpolationSurfaceHoverStore {
  private state: InterpolationSurfaceHoverState | null = null;
  private enabled = true;
  private revision = 0;
  private readonly listeners = new Set<() => void>();

  get isInspectionEnabled(): boolean {
    return this.enabled;
  }
  getState = (): InterpolationSurfaceHoverState | null => this.state;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
  /** Disabling inspection clears any visible tooltip so points under the surface stay hoverable. */
  setInspectionEnabled(enabled: boolean): void {
    if (enabled === this.enabled) return;
    this.enabled = enabled;
    if (!enabled) this.state = null;
    this.emit();
  }
  /** Returns true when the pick was consumed (the native Kepler hover should be cleared). */
  setPick(pick: InterpolationSurfacePick | null): boolean {
    if (!pick) {
      if (this.state) {
        this.state = null;
        this.emit();
      }
      return false;
    }
    const previous = this.state;
    if (previous?.surfaceId === pick.surfaceId && previous.cellIndex === pick.cellIndex) {
      // Same cell: only follow the pointer.
      this.state = {...previous, x: pick.x, y: pick.y};
    } else {
      this.state = {...pick, revision: ++this.revision};
    }
    this.emit();
    return true;
  }
  clear(): void {
    this.setPick(null);
  }
  private emit(): void {
    for (const listener of this.listeners) listener();
  }
}

export const interpolationSurfaceHover = new InterpolationSurfaceHoverStore();
