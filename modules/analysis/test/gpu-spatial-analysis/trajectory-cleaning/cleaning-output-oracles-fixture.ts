// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const POSITIONS = Float32Array.of(11, 12, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62);
export const OFFSETS = Uint32Array.of(0, 4, 6);
export const SPLIT_TIMES = Float32Array.of(2, 3, 7, 8, 20, 21);
export const VALUES = Uint32Array.of(1, 1, 2, 2, 9, 9);
// Value changes repeat the boundary fix: [0,1,2], [2,3], [4,5]. These are input copies,
// so every component is exactly representable float32 and the tolerance is zero.
export const SPLIT_POSITIONS = [11, 12, 21, 22, 31, 32, 31, 32, 41, 42, 51, 52, 61, 62];
export const SPLIT_OUTPUT_TIMES = [2, 3, 7, 7, 8, 20, 21];
export const THIN_TIMES = Float32Array.of(0, 1, 4, 6, 20, 21);
// Minimum time 3: first track keeps 0,2 then endpoint 3; second keeps endpoints 4,5.
export const THIN_POSITIONS = [11, 12, 31, 32, 41, 42, 51, 52, 61, 62];
export const THIN_OUTPUT_TIMES = [0, 4, 6, 20, 21];
export const THIN_VALIDITY = [1, 0, 1, 1, 1, 1];
