// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Validation has no tunable props; the sliders set how many rows are broken and how rows are chunked.
export const parameters: GpuSceneParameter[] = [
  {id: 'defects', label: 'Data: broken rows', min: 0, max: 0.4, step: 0.02, default: 0.12},
  {id: 'chunkCount', label: 'Data: chunks', min: 1, max: 4, step: 1, default: 3}
];
