// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// tolerance is the search's stopping tolerance in world units; bend curves the shapes.
export const parameters: GpuSceneParameter[] = [
  {id: 'tolerance', label: 'tolerance', min: 0.002, max: 0.06, step: 0.002, default: 0.004},
  {id: 'bend', label: 'Data: bend', min: 0, max: 1, step: 0.05, default: 0.6}
];
