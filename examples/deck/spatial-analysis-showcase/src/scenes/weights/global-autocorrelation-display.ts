// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {formatNumber} from './b4-format';

/** The exact observed readout, retaining half joins from asymmetric weights. */
export function formatGlobalJoinReadout(
  complete: boolean,
  counts: ArrayLike<number>,
  expectedBlackBlack: number,
  expectedBlackWhite: number
): string {
  const observed = getGlobalObservedJoinCounts(complete, counts);
  const formatJoin = (value: number) => value.toLocaleString('en-US', {maximumFractionDigits: 1});
  return observed
    ? `BB ${formatJoin(observed[0])} (E ${formatNumber(expectedBlackBlack, 0)}), BW ${formatJoin(observed[1])} (E ${formatNumber(expectedBlackWhite, 0)}), WW ${formatJoin(observed[2])}`
    : 'Unavailable (incomplete weights)';
}

type DisplayColor = readonly [number, number, number, number];
/** Tooltip for the class actually published by the display shader. */
export function getGlobalJoinTooltip(
  joinClass: number | undefined,
  complete: boolean,
  black: DisplayColor,
  white: DisplayColor
) {
  const available = complete && (joinClass === 0 || joinClass === 1);
  return {
    label: 'Join-count colour',
    value: !complete
      ? 'Unavailable (incomplete weights)'
      : !available
        ? 'No data'
        : joinClass === 1
          ? 'Black'
          : 'White',
    swatch: available ? (joinClass === 1 ? black : white) : undefined,
    emphasis: true
  };
}

/** Count the displayed codes, including all rows with no available quadrant. */
export function getGlobalQuadrantCounts(quadrants: ArrayLike<number>, islands: number): number[] {
  const counts = [0, 0, 0, 0, 0, islands];
  for (let row = 0; row < quadrants.length; row++) {
    const code = quadrants[row];
    counts[code >= 1 && code <= 4 ? code - 1 : 4]++;
  }
  return counts;
}

/** Pure legend builder shared by the scene and the GPU display regression. */
export function getGlobalQuadrantLegend(
  counts: number[] | undefined,
  colors: {
    highHigh: DisplayColor;
    lowLow: DisplayColor;
    highLow: DisplayColor;
    lowHigh: DisplayColor;
  },
  noData: DisplayColor
) {
  return {
    kind: 'categories' as const,
    id: 'quadrant-classes',
    title: 'Moran scatterplot quadrant (signs only, not tested)',
    layout: 'list' as const,
    interactive: true,
    entries: [
      {color: colors.highHigh, label: 'High-High', count: counts?.[0]},
      {color: colors.lowLow, label: 'Low-Low', count: counts?.[2]},
      {color: colors.highLow, label: 'High-Low', count: counts?.[3]},
      {color: colors.lowHigh, label: 'Low-High', count: counts?.[1]},
      {color: noData, label: 'Unavailable quadrant', count: counts?.[4]}
    ],
    note: `High or low against the mean; no significance test. Unavailable includes missing values, incomplete weights and islands. Islands: ${counts ? counts[5] : 'unavailable'}.`
  };
}

/** An incomplete graph has no observed join counts, even when its storage words are zero. */
export function getGlobalObservedJoinCounts(
  complete: boolean,
  counts: ArrayLike<number>
): number[] | undefined {
  return complete ? [counts[0] / 2, counts[1] / 2, counts[2] / 2] : undefined;
}

/** Matches the display shader's one-based quadrant codes; zero denotes unavailable. */
export function getGlobalMoranQuadrant(
  value: number,
  lag: number,
  mean: number,
  complete: boolean
): number {
  if (!complete || ![value, lag, mean].every(Number.isFinite)) return 0;
  return value >= mean ? (lag >= mean ? 1 : 4) : lag >= mean ? 2 : 3;
}

export const CLASSIFY_BODY = /* wgsl */ `
  let included = mask[maskOffset + index] != 0u;
  let value = x[xOffset + index];
  let lagged = lag[lagOffset + index];
  let mean = summary[summaryOffset + ${GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.mean}u];
  let finite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u &&
    (bitcast<u32>(lagged) & 0x7f800000u) != 0x7f800000u &&
    (bitcast<u32>(mean) & 0x7f800000u) != 0x7f800000u;
  if (!included || !finite) {
    quadrant[quadrantOffset + index] = 0u;
    joinClass[joinClassOffset + index] = 0xffffffffu;
    var nanBits = 0x7fc00000u;
    binary[binaryOffset + index] = bitcast<f32>(nanBits);
    return;
  }
  // Moran scatterplot quadrant of (x - mean, lag - mean): 1 HH, 2 LH, 3 LL, 4 HL.
  let high = value >= mean;
  let neighborsHigh = lagged >= mean;
  var quadrantValue = 3u;
  if (high && neighborsHigh) { quadrantValue = 1u; }
  else if (high) { quadrantValue = 4u; }
  else if (neighborsHigh) { quadrantValue = 2u; }
  quadrant[quadrantOffset + index] = quadrantValue;
  let black = value > threshold[thresholdOffset];
  joinClass[joinClassOffset + index] = select(0u, 1u, black);
  binary[binaryOffset + index] = select(0.0, 1.0, black);`;
