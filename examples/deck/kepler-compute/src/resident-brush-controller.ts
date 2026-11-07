// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {getBrushingRadiusMeters, type BrushInteractionConfig} from './native-brushing-support';
import {ResidentBrushSelection} from './resident-brush-selection';

type BrushStoreState = {
  keplerGl?: Record<
    string,
    | {
        visState?: {
          interactionConfig?: {brush?: BrushInteractionConfig};
          mousePos?: {coordinate?: readonly number[] | null};
        };
      }
    | undefined
  >;
};

export type ResidentBrushSource = {
  datasetId: string;
  device: Device;
  /** Borrowed resident point positions (longitude, latitude); never read back. */
  positions: GPUVector<'float32x2'>;
  /**
   * Links the active brush mask into the dataset's selection scopes (table, charts, legends), or
   * unlinks it with null when the brush is off. Called after the mask is recomputed.
   */
  linkBrushMask?(mask: GPUVector<'uint32'> | null): void;
};

export type ResidentBrushControllerProps = {
  store: {getState(): unknown; subscribe(listener: () => void): () => void};
  instanceId: string;
  /** Every brushable resident dataset with point positions; empty while no runtime exists or the device is lost. */
  getSources(): readonly ResidentBrushSource[];
};

type BrushEntry = {selection: ResidentBrushSelection; positions: GPUVector<'float32x2'>};

/**
 * Drives one GPU brush selection per brushable resident dataset from Kepler's brush interaction
 * config and pointer coordinate. Pointer movement is coalesced to one GPU update per animation
 * frame. When the brush is turned off or the pointer leaves, each mask is recomputed empty and
 * unlinked so linked consumers never see stale rows.
 */
export function createResidentBrushController(props: ResidentBrushControllerProps) {
  const entries = new Map<string, BrushEntry>();
  let lastKey = '';
  let frame = 0;
  const listeners = new Set<() => void>();

  const readInputs = () => {
    const visState = (props.store.getState() as BrushStoreState).keplerGl?.[props.instanceId]
      ?.visState;
    const brush = visState?.interactionConfig?.brush;
    // Native brushingEnabled gates the brush; the radius alone must not select rows.
    const radiusMeters = brush?.enabled ? getBrushingRadiusMeters(brush) : 0;
    const coordinate = visState?.mousePos?.coordinate;
    return {
      radiusMeters,
      center:
        radiusMeters > 0 && coordinate && coordinate.length >= 2
          ? ([coordinate[0], coordinate[1]] as const)
          : null
    };
  };

  const releaseEntry = (datasetId: string, source?: ResidentBrushSource) => {
    const entry = entries.get(datasetId);
    if (!entry) return;
    source?.linkBrushMask?.(null);
    entry.selection.destroy();
    entries.delete(datasetId);
  };

  const update = () => {
    frame = 0;
    const sources = props.getSources();
    const liveIds = new Set(sources.map(source => source.datasetId));
    for (const datasetId of [...entries.keys()])
      if (!liveIds.has(datasetId)) releaseEntry(datasetId);
    const {center, radiusMeters} = readInputs();
    const key = center ? `${center[0]},${center[1]},${radiusMeters}` : '';
    const keyChanged = key !== lastKey;
    lastKey = key;
    for (const source of sources) {
      let entry = entries.get(source.datasetId);
      if (entry && entry.positions !== source.positions) {
        releaseEntry(source.datasetId, source);
        entry = undefined;
      }
      if (!entry && !center) continue;
      if (!entry) {
        const selection = new ResidentBrushSelection({
          device: source.device,
          datasetId: source.datasetId,
          positions: source.positions
        });
        selection.subscribe(() => {
          for (const listener of listeners) listener();
        });
        entry = {selection, positions: source.positions};
        entries.set(source.datasetId, entry);
      } else if (!keyChanged) continue;
      entry.selection.setBrush(center, radiusMeters);
      source.linkBrushMask?.(entry.selection.isActive ? entry.selection.mask : null);
    }
  };

  const unsubscribe = props.store.subscribe(() => {
    if (!frame) frame = requestAnimationFrame(update);
  });

  return {
    /** The live brush scope of one dataset, or null until the brush is first used there. */
    getSelection(datasetId: string): ResidentBrushSelection | null {
      return entries.get(datasetId)?.selection ?? null;
    },
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    destroy(): void {
      unsubscribe();
      if (frame) cancelAnimationFrame(frame);
      const sources = new Map(props.getSources().map(source => [source.datasetId, source]));
      for (const datasetId of [...entries.keys()]) releaseEntry(datasetId, sources.get(datasetId));
      listeners.clear();
    }
  };
}
