// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Brent cycle/stagnation detector: two fixed float64 vectors, no iteration history. @internal */
export class SpatialInteractionCycleDetector {
  private readonly previous: Float64Array;
  private readonly checkpoint: Float64Array;
  private power = 1;
  private length = 0;

  constructor(values: readonly number[]) {
    this.previous = Float64Array.from(values);
    this.checkpoint = Float64Array.from(values);
  }

  observe(values: readonly number[], rows: readonly number[]): boolean {
    let samePrevious = true;
    let sameCheckpoint = true;
    for (let index = 0; index < rows.length; index++) {
      const value = values[rows[index]];
      samePrevious &&= value === this.previous[index];
      sameCheckpoint &&= value === this.checkpoint[index];
      this.previous[index] = value;
    }
    this.length++;
    if (samePrevious || sameCheckpoint) return true;
    if (this.length === this.power) {
      this.checkpoint.set(this.previous);
      this.power *= 2;
      this.length = 0;
    }
    return false;
  }
}
