// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Readout units. The operations run in world units and playback seconds (a UTC day plays in 600 s,
// so one playback second is 144 s of vessel time); readouts show what a person would say.

const VESSEL_SECONDS_PER_PLAYBACK_SECOND = 144;
const METERS_PER_SECOND_PER_KNOT = 0.514444;

/** Meters, or kilometers from one kilometer up. */
export function formatDistance(worldUnits: number, metersPerUnit: number): string {
  const meters = worldUnits * metersPerUnit;
  if (meters >= 1000) return `${(meters / 1000).toFixed(meters >= 10000 ? 0 : 1)} km`;
  return `${Math.round(meters / 10) * 10} m`;
}

/** Minutes, or hours and minutes, of vessel time. */
export function formatDuration(playbackSeconds: number): string {
  const minutes = Math.round((playbackSeconds * VESSEL_SECONDS_PER_PLAYBACK_SECOND) / 60);
  if (minutes < 60) return `${minutes} min`;
  const rest = minutes % 60;
  return rest ? `${Math.floor(minutes / 60)} h ${rest} min` : `${minutes / 60} h`;
}

/** Knots from world units per playback second. */
export function formatSpeed(worldUnitsPerPlaybackSecond: number, metersPerUnit: number): string {
  const knots =
    (worldUnitsPerPlaybackSecond * metersPerUnit) /
    VESSEL_SECONDS_PER_PLAYBACK_SECOND /
    METERS_PER_SECOND_PER_KNOT;
  return `${knots.toFixed(knots < 10 ? 1 : 0)} kn`;
}
