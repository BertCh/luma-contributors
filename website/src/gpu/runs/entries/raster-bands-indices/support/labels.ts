// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Display helpers for label rasters: graph-coloured fills, the hovered-patch highlight and the
// categories that keep background (label 0) muted.

import type {RunBuffer} from '../../../define-run';
import type {DemoTheme} from '../../../../demos/kit';
import {toUint32Array} from './gestures';

export const HIGHLIGHT = [1, 1, 1, 0.55] as const;
export const HIGHLIGHT_CATEGORIES = [[0, 0, 0, 0], HIGHLIGHT] as const;

/** Muted background, then seven categorical colours for graph-coloured labels. */
export function getLabelCategories(theme: DemoTheme) {
  return [theme.muted, ...theme.categories.slice(0, 7)];
}

/** Cell count per label of a label raster. */
export function countLabelCells(labels: Uint32Array): Uint32Array {
  let maximum = 0;
  for (const label of labels) maximum = Math.max(maximum, label);
  const counts = new Uint32Array(maximum + 1);
  for (const label of labels) counts[label]++;
  return counts;
}

/** 1 for cells of `label`, else 0: the hovered-patch highlight. */
export function maskLabel(labels: Uint32Array, label: number): Uint32Array {
  return Uint32Array.from(labels, value => (value === label && label > 0 ? 1 : 0));
}

/** Reads a uint32 GPU buffer back to the CPU. */
export async function readUint32(buffer: RunBuffer): Promise<Uint32Array> {
  return toUint32Array(await buffer.buffer.readAsync());
}

/** Reads a float32 GPU buffer back to the CPU. */
export async function readFloat32(buffer: RunBuffer): Promise<Float32Array> {
  const bytes = await buffer.buffer.readAsync();
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  );
}
