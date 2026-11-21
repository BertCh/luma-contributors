// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPURasterFixedPointValues} from '../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {expect} from 'vitest';

export type MomentRoute = 'primitive' | 'streaming' | 'calibration';

export function decodeMomentProbe(
  words: readonly number[],
  route: MomentRoute,
  unitExponent: number
): number {
  expect(words).toHaveLength(route === 'streaming' ? 3 : 9);
  const offset = route === 'streaming' ? 0 : 6;
  const decoded = getGPURasterFixedPointValues(
    Uint32Array.from(words.slice(offset, offset + 3)),
    unitExponent
  );
  expect(decoded).toHaveLength(1);
  return decoded[0];
}
