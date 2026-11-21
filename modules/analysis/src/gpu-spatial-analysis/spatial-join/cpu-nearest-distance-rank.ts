// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared GPU-compatible float32 squared-distance ranking for planar CPU joins. @internal */
export function createCPUFeatureDistanceRank(): (distance: number) => number {
  const rankFloat = new Float32Array(1),
    rankBits = new Uint32Array(rankFloat.buffer);
  const getDistanceRank = (distance: number) => {
    if (distance === 0) return 0;
    if (!Number.isFinite(distance)) return 0xff7ffffd;
    const squared = distance * distance;
    let exponent = Math.floor(Math.log2(squared)) + 1;
    rankFloat[0] = squared / 2 ** exponent;
    if (rankFloat[0] === 1) {
      rankFloat[0] = 0.5;
      exponent++;
    }
    if (exponent > 256 || (exponent === 256 && rankFloat[0] > 0.9999998807907104))
      return 0xff7ffffd;
    if (exponent < -251) return 0x00800000;
    let word = (exponent + 252) * 0x800000 + (rankBits[0] & 0x7fffff);
    if (word >= 0x7f800000) word += 0x01000000;
    return Math.min(word, 0xff7ffffc);
  };

  return getDistanceRank;
}
