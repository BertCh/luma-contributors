// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_MAX_P_STATUS} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** One successful batch can reassign several independent rows. */
export function formatMaxPImprovementRounds(status: ArrayLike<number>): string {
  return `successful improvement rounds ${status[GPU_MAX_P_STATUS.improvementIterationCount]}`;
}
