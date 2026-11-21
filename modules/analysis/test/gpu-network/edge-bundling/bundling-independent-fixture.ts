// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent NumPy real64 reference: splat max(0,1-r^2) with 1/65536 quanta, bilinearly
// sample its central gradient, advect by radius*0.1, resample equal arc lengths, then smooth
// by 0.2 of the neighboring midpoint displacement. Repeat eight radii 0.15*0.8^iteration.
// Work box [60,60,880]. The derivation script is retained in the stream's scratch folder as
// bundling-reference.py. These literals do not come from the TypeScript oracle or GPU output.
export const BUNDLING_INDEPENDENT_PATHS = [
  100, 450, 298.71865274, 484.65367933, 500, 495.7817279, 701.28134726, 484.65367933, 900, 450, 100,
  550, 298.71865274, 515.34632067, 500, 504.2182721, 701.28134726, 515.34632067, 900, 550
];

export const BUNDLING_INDEPENDENT_SETTINGS = {
  kernelRadius: 0.15,
  lambda: 0.8,
  smoothing: 0.2,
  stepScale: 0.1
};
