// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * float32 bit patterns of `output.costs` from `GPURasterWalkingCost` exactly as it was at commit
 * 74dddcf5 (before the `parameters` rewrite), on the inputs of
 * `raster-walking-cost-parameters.spec.ts`: the 7 x 5 elevation and friction rasters, source cell
 * 15, a 3 m grid at (500000, 4600000), `maximumIterations: 64`. Captured once on the headless
 * WebGPU test device (Chromium, Metal) from `git show 74dddcf5:modules/analysis/src/raster-analysis/gpu-raster-walking-cost.ts`
 * run against the current helpers (their changes since 74dddcf5 are additive). Both runs reported
 * `converged = 1`. Do not regenerate from the current kernel: the point is the old output.
 */
export const WALKING_COST_74DDDCF5_COST_BITS = {
  /** `{frictionWeight: 1, model: {kind: 'tobler'}}`. */
  tobler: [
    0x4104e523, 0x41d3147b, 0x42626a41, 0x4294c6ab, 0x42be4d3e, 0x42db0f6e, 0x42f6207f, 0x407281e9,
    0x41b50afc, 0x42161faa, 0x425005fe, 0x4281faa4, 0x42c5fac1, 0x42ecb050, 0x412721c1, 0x00000000,
    0x4157742b, 0x4228045a, 0x426b0297, 0x42c8ebe6, 0x430154ba, 0x421510b6, 0x4178b727, 0x40a6b2e2,
    0x4236c8eb, 0x428a60e0, 0x42d2b93c, 0x43036ad9, 0x4233e302, 0x4193e282, 0x416db042, 0x425c6011,
    0x428fb8bf, 0x42bac342, 0x42da671f
  ],
  /** `{frictionWeight: 0.5, model: {kind: 'grass', coefficients: [0.72, 6, 1.9998, -1.9998], downhillSlopeThreshold: -0.2}}`. */
  grass: [
    0x41098c95, 0x41cfc9db, 0x4243f96e, 0x4284dec6, 0x42a08020, 0x42b85eea, 0x42d438c5, 0x4091b054,
    0x41a7c6b2, 0x42086e9e, 0x42384f3a, 0x426dff88, 0x429b23c8, 0x42ca8280, 0x40afa75f, 0x00000000,
    0x414fc7f7, 0x41f230e2, 0x4254ca02, 0x42a0f2f2, 0x42d98b6c, 0x41531ea8, 0x40f1105c, 0x40df12d0,
    0x42072e5b, 0x4278c17e, 0x42b0949c, 0x42dab2e9, 0x41914799, 0x4132efb2, 0x4186e9fd, 0x42290005,
    0x42803e81, 0x42a0b411, 0x42b9c449
  ]
} as const;
