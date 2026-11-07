// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {
    id: 'maximumCircumradius',
    label: 'maximumCircumradius',
    min: 0.02,
    max: 0.4,
    step: 0.01,
    default: 0.09
  }
];
