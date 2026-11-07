// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

export type ResidentChannelLegendScale = 'linear' | 'sqrt' | 'log';
export type ResidentChannelLegendTick = {value: number; size: number};
export type ResidentChannelLegendModel =
  | {status: 'ready'; ticks: readonly ResidentChannelLegendTick[]}
  | {status: 'unsupported' | 'empty'; reason: string};

function transformValue(scale: ResidentChannelLegendScale, value: number): number {
  return scale === 'sqrt' ? Math.sqrt(value) : scale === 'log' ? Math.log(value) : value;
}

/**
 * Evenly spaced size/radius/height legend ticks over a domain, mapped through the channel scale
 * into the visual range. The domain must be GPU-derived by the caller to be labelled as such.
 */
export function getResidentChannelLegendModel(options: {
  scale: string;
  domain: readonly number[] | null | undefined;
  range: readonly [number, number];
  tickCount?: number;
}): ResidentChannelLegendModel {
  const {scale, domain, range} = options;
  if (scale !== 'linear' && scale !== 'sqrt' && scale !== 'log')
    return {status: 'unsupported', reason: `Legend for ${scale} scale is not available.`};
  if (!domain || domain.length < 2 || !domain.every(Number.isFinite))
    return {status: 'empty', reason: 'No finite values in the size domain.'};
  const minimum = domain[0];
  const maximum = domain[domain.length - 1];
  if ((scale === 'log' && minimum <= 0) || (scale === 'sqrt' && minimum < 0))
    return {status: 'unsupported', reason: 'The domain is not valid for this scale.'};
  const tickCount = clamp(options.tickCount ?? 5, 2, 10);
  const lower = transformValue(scale, minimum);
  const upper = transformValue(scale, maximum);
  const ticks = Array.from({length: tickCount}, (_, index) => {
    const value = minimum + ((maximum - minimum) * index) / (tickCount - 1);
    const fraction = upper === lower ? 0 : (transformValue(scale, value) - lower) / (upper - lower);
    return {value, size: range[0] + (range[1] - range[0]) * fraction};
  });
  return {status: 'ready', ticks};
}
