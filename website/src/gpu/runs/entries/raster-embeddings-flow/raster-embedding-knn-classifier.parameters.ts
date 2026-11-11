// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'neighborCount', label: 'neighborCount', min: 1, max: 15, step: 2, default: 5},
  {
    id: 'perClass',
    label: 'Data: labelled pixels per class',
    min: 0,
    max: 2,
    step: 1,
    default: 1,
    options: ['3', '8', '20']
  },
  {
    id: 'shown',
    label: 'Output shown',
    min: 0,
    max: 1,
    step: 1,
    default: 0,
    options: ['labels', 'confidence']
  }
];
