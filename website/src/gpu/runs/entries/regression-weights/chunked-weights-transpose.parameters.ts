// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The transpose has no tunable props; k shapes the directed input, chunks how its rows are split.
export const parameters: GpuSceneParameter[] = [
  {id: 'k', label: 'Data: k', min: 2, max: 8, step: 1, default: 4},
  {id: 'chunkCount', label: 'Data: chunks', min: 1, max: 4, step: 1, default: 3}
];
