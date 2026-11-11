// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Highlighting every cell of one class: the pointer picks a class, a render-only mask (written by
// the interaction) marks the other cells, and a translucent layer dims them.

import type {RasterRunLayer, RunBuffer} from '../../../define-run';

type MaskState = {classes: ArrayLike<number> | null; selected: number};

/** A fresh cache for {@link writeClassMask}; keep one in the run's inputs. */
export function createMaskState(): MaskState {
  return {classes: null, selected: -1};
}

/**
 * Writes 1 over every cell outside class `selected` (all 0 when `selected` is negative). Skips the
 * write when neither the class array nor the selection changed.
 */
export function writeClassMask(
  mask: RunBuffer<'float32'>,
  classes: ArrayLike<number>,
  selected: number,
  state: MaskState
): void {
  if (state.classes === classes && state.selected === selected) return;
  state.classes = classes;
  state.selected = selected;
  const values = new Float32Array(classes.length);
  if (selected >= 0) {
    for (let index = 0; index < classes.length; index++) {
      values[index] = classes[index] === selected ? 0 : 1;
    }
  }
  mask.write(values);
}

/** The layer that dims the cells the mask marks, by default with a translucent grey. */
export function drawClassDimmer(
  terrain: {columns: number; rows: number},
  mask: RunBuffer,
  dimColor: readonly [number, number, number, number] = [0.45, 0.47, 0.52, 0.6]
): RasterRunLayer {
  return {
    layer: 'raster',
    values: mask,
    columns: terrain.columns,
    rows: terrain.rows,
    ramp: 'categorical',
    categories: [[0, 0, 0, 0], dimColor]
  };
}
