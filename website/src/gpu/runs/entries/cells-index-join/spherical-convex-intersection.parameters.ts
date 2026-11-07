// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// The polygons of each pair are rewritten per frame: `separation` moves the pair apart and
// `rotation` turns the second polygon; `sides` changes the vertex count (compile time).
export const parameters: GpuSceneParameter[] = [
  {id: 'sides', label: 'Data: sides', min: 3, max: 8, step: 1, default: 5},
  {id: 'separation', label: 'Data: separation', min: 0, max: 0.34, step: 0.01, default: 0.14},
  {
    id: 'rotation',
    label: 'Data: rotation',
    min: 0,
    max: 1.2,
    step: 0.02,
    default: 0.5,
    unit: ' rad'
  }
];
