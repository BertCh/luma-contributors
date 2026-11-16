// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Reserve the uint32 maximum for a saturated count strictly above every accepted capacity.
export const TRAJECTORY_MAXIMUM_CANDIDATE_CAPACITY = 0xfffffffe;

/**
 * Reserves a trajectory candidate/hit slot without wrapping uint32. Expects `state` to be
 * bound as array<atomic<u32>> and the counter at stateOffset. The count saturates at uint32
 * maximum, which remains above every accepted candidate capacity.
 * A saturated reservation returns the sentinel and must never write a candidate slot.
 */
export const TRAJECTORY_SATURATED_COUNT_WGSL = /* wgsl */ `
// The maximum is a sentinel: counts beyond uint32 must not wrap back into an output prefix.
const TRAJECTORY_COUNT_MAXIMUM: u32 = 0xffffffffu;
fn reserveTrajectoryCount() -> u32 {
  var currentCount = atomicLoad(&state[stateOffset]);
  loop {
    if (currentCount == TRAJECTORY_COUNT_MAXIMUM) { return currentCount; }
    let comparison = atomicCompareExchangeWeak(&state[stateOffset], currentCount, currentCount + 1u);
    if (comparison.exchanged) { return currentCount; }
    currentCount = comparison.old_value;
  }
}
`;
