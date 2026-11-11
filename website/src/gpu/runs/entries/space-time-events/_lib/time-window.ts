// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The time axis of the lightning runs: the 18-hour observation period spans the canvas width, and
// the glyphs show the slice time (the inner 90% of the width, as the pointer pad maps it) and its temporal bandwidth as a bar along the bottom.

import type {RunGlyph} from '../../../interaction';

/** Hours of the GOES flash sample, from 2024-05-21 12:00 UTC. */
export const OBSERVATION_HOURS = 18;
const AXIS_Y = 0.025;

/** Gaussian weight of every flash time against the slice time, for the flash layer. */
export function getTimeWeights(
  hours: Float32Array,
  sliceTime: number,
  bandwidth: number
): Float32Array {
  return Float32Array.from(hours, hour => Math.exp(-0.5 * ((hour - sliceTime) / bandwidth) ** 2));
}

/** Formats elapsed hours as the UTC clock time. */
export function formatClock(hours: number): string {
  const total = Math.round((12 + hours) * 60) % (24 * 60);
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

/** Axis, ±bandwidth bar and slice tick at the bottom, plus a readout of the window. */
export function createTimeWindowGlyphs(
  hours: Float32Array,
  sliceTime: number,
  bandwidth: number,
  width: number,
  label = 'flashes'
): {glyphs: RunGlyph[]; readout: string} {
  const toX = (hour: number) =>
    Math.min(width, Math.max(0, (hour / OBSERVATION_HOURS) * width * 0.9 + width * 0.05));
  let flashCount = 0;
  for (const hour of hours) if (Math.abs(hour - sliceTime) <= bandwidth) flashCount++;
  const glyphs: RunGlyph[] = [
    {shape: 'line', from: [toX(0), AXIS_Y], to: [toX(OBSERVATION_HOURS), AXIS_Y], dashed: true},
    {
      shape: 'box',
      minimum: [toX(sliceTime - bandwidth), AXIS_Y - 0.008],
      maximum: [toX(sliceTime + bandwidth), AXIS_Y + 0.008]
    },
    {shape: 'line', from: [toX(sliceTime), AXIS_Y - 0.02], to: [toX(sliceTime), AXIS_Y + 0.02]}
  ];
  return {
    glyphs,
    readout: `${formatClock(sliceTime)} UTC ± ${bandwidth} h · ${flashCount} ${label}`
  };
}
