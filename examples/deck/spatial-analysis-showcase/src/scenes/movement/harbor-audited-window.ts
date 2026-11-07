// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {createTrackSet, loadVesselTracks} from './b12-tracks';

export const HARBOR_AUDITED_SOURCE_PATHS = [221, 370, 509, 562, 595, 712, 726, 735] as const;
export const HARBOR_AUDITED_WINDOW_START = 14 * 3600;
export const HARBOR_AUDITED_WINDOW_END = 15 * 3600;

/** Real selected path fixes restricted to their observed support in the audited UTC hour. */
export function loadHarborAuditedWindow(dataset: LoadedDataset) {
  const source = loadVesselTracks(dataset);
  const selectedOffsets = [0];
  const selectedPositions: number[] = [];
  const selectedLngLat: number[] = [];
  const selectedTimes: number[] = [];
  const categories: number[] = [];
  const mmsi: number[] = [];
  const maximumObservedGaps: number[] = [];
  const firstFixTimes: number[] = [];
  const sourceRows: number[] = [];
  for (const sourceTrack of HARBOR_AUDITED_SOURCE_PATHS) {
    const start = source.offsets[sourceTrack];
    const end = source.offsets[sourceTrack + 1];
    const rowStart = selectedTimes.length;
    let maximumGap = 0;
    for (let row = start; row < end; row++) {
      const time = source.timestamps[row];
      if (time < HARBOR_AUDITED_WINDOW_START || time > HARBOR_AUDITED_WINDOW_END) continue;
      if (selectedTimes.length > rowStart)
        maximumGap = Math.max(
          maximumGap,
          time - HARBOR_AUDITED_WINDOW_START - selectedTimes[selectedTimes.length - 1]
        );
      selectedPositions.push(source.positions[row * 2], source.positions[row * 2 + 1]);
      selectedLngLat.push(source.lngLat[row * 2], source.lngLat[row * 2 + 1]);
      selectedTimes.push(time - HARBOR_AUDITED_WINDOW_START);
    }
    if (selectedTimes.length - rowStart < 2)
      throw new Error(
        `Audited AIS path ${sourceTrack} has insufficient fixes in the selected UTC hour`
      );
    selectedOffsets.push(selectedTimes.length);
    firstFixTimes.push(selectedTimes[rowStart]);
    maximumObservedGaps.push(maximumGap);
    categories.push(source.category[sourceTrack]);
    mmsi.push(source.mmsi[sourceTrack]);
    sourceRows.push(sourceTrack);
  }
  const tracks = createTrackSet({
    offsets: Uint32Array.from(selectedOffsets),
    positions: Float32Array.from(selectedPositions),
    lngLat: Float32Array.from(selectedLngLat),
    timestamps: Float32Array.from(selectedTimes),
    origin: source.origin,
    analysisFrame: source.analysisFrame,
    spatialContext: source.spatialContext,
    project: source.project,
    unproject: source.unproject,
    drawInDegrees: false
  });
  return {
    tracks,
    categories: Uint32Array.from(categories),
    mmsi: Uint32Array.from(mmsi),
    maximumObservedGaps: Uint32Array.from(maximumObservedGaps),
    firstFixTimes: Float32Array.from(firstFixTimes),
    sourceRows: Uint32Array.from(sourceRows)
  };
}
