// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type FacilityCompactionRound = 'active' | 'invalid' | 'stopped' | 'disarmed';

/** Shared submission inputs for the GPU resubmission test and host gate discrimination probe. */
export function getFacilityCompactionRoundInputs(mode: FacilityCompactionRound) {
  return {
    status: Uint32Array.of(0, 0, 0, mode === 'invalid' || mode === 'disarmed' ? 1 : 0, 0, 0, 0),
    state: Uint32Array.of(mode === 'stopped' ? 1 : 0, 0, 0),
    dispatch: Uint32Array.of(mode === 'disarmed' ? 0 : 1, 1, 1)
  };
}
