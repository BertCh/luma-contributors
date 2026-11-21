// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {WeightsRig} from '../spatial-weights/spatial-weights-harness';

/** Input loss ports shared by both same-graph partition recovery fixtures. */
export function createPartitionLossPorts(rig: WeightsRig) {
  return {
    overflow: rig.input(Uint32Array.of(0), 'uint32', 1),
    candidateOverflow: rig.input(Uint32Array.of(0), 'uint32', 1)
  };
}
