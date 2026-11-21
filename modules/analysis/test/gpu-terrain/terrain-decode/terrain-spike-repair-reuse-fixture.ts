// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Per-frame inputs shared with the device reuse regression; all heights use a 256-unit step. */
export function getSpikeRepairReuseFrames() {
  const first = new Float32Array(25).fill(100);
  for (const cell of [6, 7, 11, 12]) first[cell] = 356;
  const second = new Float32Array(25).fill(100);
  second[6] = -156;
  second[11] = 12345;
  const secondValidity = new Uint32Array(25).fill(1);
  secondValidity[11] = 0;
  return {first, second, secondValidity};
}
