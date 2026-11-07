// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// nodeCapacity bounds how many depressions the tree may hold; stages sets how many elevation
// stages the area and volume curves are measured at. Both are compile-time props.
export const parameters: GpuSceneParameter[] = [
  {id: 'nodeCapacity', label: 'nodeCapacity', min: 8, max: 256, step: 8, default: 256},
  {id: 'stages', label: 'stages', min: 2, max: 12, step: 1, default: 6}
];
