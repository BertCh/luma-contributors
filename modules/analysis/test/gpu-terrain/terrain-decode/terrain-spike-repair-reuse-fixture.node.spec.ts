// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getSpikeRepairReuseFrames} from './terrain-spike-repair-reuse-fixture';

function countShiftedSeams(values: Float32Array, validity: Uint32Array) {
  // These fixtures have only a flat base and a +/-256 step. The shifted pixels form one
  // component; count its border directly from the input, independently of the GPU labeler.
  const members = Array.from(values.keys()).filter(cell => validity[cell] && values[cell] !== 100);
  let seams = 0;
  for (const cell of members) {
    const column = cell % 5;
    const row = Math.floor(cell / 5);
    for (const [columnOffset, rowOffset] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1]
    ]) {
      const otherColumn = column + columnOffset;
      const otherRow = row + rowOffset;
      if (otherColumn < 0 || otherColumn >= 5 || otherRow < 0 || otherRow >= 5) continue;
      const other = otherRow * 5 + otherColumn;
      if (validity[other] && values[other] === 100) seams++;
    }
  }
  return {label: Math.min(...members), count: members.length, seams};
}

it('the reuse fixture detects a retained seam counter under the same shifted label (B2, RT-139)', () => {
  const {first, second, secondValidity} = getSpikeRepairReuseFrames();
  const previous = countShiftedSeams(first, new Uint32Array(25).fill(1));
  const current = countShiftedSeams(second, secondValidity);
  // Interior 2x2: eight edges, label 6. Shrunk cell 6: three valid neighbours after masking 11.
  expect(previous).toEqual({label: 6, count: 4, seams: 8});
  expect(current).toEqual({label: 6, count: 1, seams: 3});
  const retainedSeams = current.seams + (previous.label === current.label ? previous.seams : 0);
  expect(retainedSeams).toBe(11);
  // With a reset, all three votes agree. A mutant that omits only the seam reset fails the
  // 80% threshold: 3/11 < 0.8. The device test must then fail its required repaired-height check.
  expect(current.seams >= 0.8 * current.seams).toBe(true);
  expect(current.seams >= 0.8 * retainedSeams).toBe(false);
});
