// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import acquisition from '../../../../data/recipes/dixie-ndvi.json';

const scenes = acquisition.slices.map(slice =>
  Uint8Array.from(atob(slice.values), value => value.charCodeAt(0))
);

/** Valid block-mean NDVI, aligned to a normalized x-right/y-up world position. */
export function getVegetationIndex(x: number, y: number, slice: number): number {
  const column = Math.max(0, Math.min(acquisition.width - 1, Math.floor(x * acquisition.width)));
  const row = Math.max(
    0,
    Math.min(acquisition.height - 1, Math.floor((1 - y) * acquisition.height))
  );
  const code = scenes[slice][row * acquisition.width + column];
  return code ? code / 127.5 - 1 : Number.NaN;
}
