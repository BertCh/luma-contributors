// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumVerticesPerGroup',
    label: 'maximumVerticesPerGroup',
    min: 4,
    max: 24,
    step: 1,
    default: 24
  },
  {id: 'groupCount', label: 'groupCount', min: 1, max: 4, step: 1, default: 4}
];
