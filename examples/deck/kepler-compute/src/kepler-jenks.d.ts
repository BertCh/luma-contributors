// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Aliased in vite.config.ts to the pinned `@kepler.gl/utils` jenks module (not in package exports). */
declare module '@kepler.gl/utils/jenks' {
  export default function naturalBreaks(values: ArrayLike<number>, classCount: number): number[];
}
