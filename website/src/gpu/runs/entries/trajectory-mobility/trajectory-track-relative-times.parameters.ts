// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// GPUTrajectoryTrackRelativeTimes has no per-frame prop. The slider moves the input: the calendar
// year of the Int64 epoch-millisecond timestamps, which sets how coarse a plain float32 would be.
export const parameters: GpuSceneParameter[] = [
  {id: 'year', label: 'Data: year', min: 1970, max: 2030, step: 1, default: 2023}
];
