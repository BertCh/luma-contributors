// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerGeoJSONTripGPUVectors} from './geometry-preparation';

import {clamp} from '@math.gl/core';

/** Per-vertex or per-trip channels that the trip runtime gathers into owned outputs. */
export type TripOutputChannels = {
  selectionMask: GPUVector | undefined;
  colors?: GPUVector;
  widths?: GPUVector;
  orientation?: readonly (GPUVector | undefined)[];
};

/**
 * Exact bytes of the owned trip outputs: one word per trip for each gathered channel, member
 * flags, same-layout filtered paths and times, and one word per member row for each marker
 * channel (colors plus every configured orientation field).
 */
export function estimateTripOutputBytes(
  trips: KeplerGeoJSONTripGPUVectors,
  channels: TripOutputChannels
): number {
  const {getPath, sourceRowIds, membership} = trips;
  const tripBytes = sourceRowIds.data.reduce(
    (total, chunk) => total + Math.max(4, chunk.length * 4),
    0
  );
  const memberBytes =
    membership?.memberRows.data.reduce(
      (total, chunk) => total + Math.max(4, chunk.length * 4),
      0
    ) ?? 0;
  // Filtered paths and times duplicate source vertex storage so source buffers stay untouched.
  const filteredBytes = membership
    ? getPath.data.reduce(
        (total, chunk) =>
          total + Math.max(4, chunk.valueLength * 12) + Math.max(4, chunk.valueLength * 4),
        0
      )
    : 0;
  const orientationCount = (channels.orientation ?? []).filter(Boolean).length;
  const markerBytes = membership
    ? memberBytes * (Number(Boolean(channels.colors)) + orientationCount)
    : 0;
  const gatheredCount = [channels.selectionMask, channels.colors, channels.widths].filter(
    Boolean
  ).length;
  return tripBytes * gatheredCount + memberBytes + filteredBytes + markerBytes;
}

/**
 * Working budget for one trip runtime. The configured per-style budget stays the default; a
 * trip whose exact outputs plus a quarter of transient headroom do not fit grows to what it needs,
 * limited to the registry bytes still available, so large grouped trips keep their marker channels
 * instead of falling back to trails. Returns the base budget when growth is not allowed.
 */
export function resolveTripWorkingBudget(
  baseBudgetBytes: number,
  estimatedOutputBytes: number,
  availableBytes: number | undefined
): number {
  const requiredBytes = estimatedOutputBytes + Math.ceil(estimatedOutputBytes / 4);
  if (availableBytes === undefined || requiredBytes <= baseBudgetBytes) return baseBudgetBytes;
  return clamp(Math.floor(availableBytes), baseBudgetBytes, requiredBytes);
}
