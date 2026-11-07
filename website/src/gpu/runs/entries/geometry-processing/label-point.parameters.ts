// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// initialGridSize and refinementRounds are the operation's search props; bend curves the shapes.
export const parameters: GpuSceneParameter[] = [
  {id: 'initialGridSize', label: 'initialGridSize', min: 2, max: 10, step: 1, default: 3},
  {id: 'refinementRounds', label: 'refinementRounds', min: 0, max: 4, step: 1, default: 3},
  {id: 'bend', label: 'Data: bend', min: 0, max: 1, step: 0.05, default: 0.6}
];
