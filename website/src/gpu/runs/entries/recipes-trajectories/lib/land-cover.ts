// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import cover from '../../../../data/recipes/hyde-park-cover.json';

const classes = Uint8Array.from(atob(cover.classes), character => character.charCodeAt(0));
const built = Uint8Array.from(atob(cover.builtFraction), character => character.charCodeAt(0));

function sample(values: Uint8Array, width: number, height: number, x: number, y: number): number {
  const column = Math.max(0, Math.min(width - 1, Math.floor(x * width)));
  const row = Math.max(0, Math.min(height - 1, Math.floor((1 - y) * height)));
  return values[row * width + column];
}

export function getCoverClass(x: number, y: number): number {
  return sample(classes, cover.width, cover.height, x, y);
}

export function getBuiltFraction(x: number, y: number): number {
  return sample(built, 64, 64, x, y) / 255;
}
