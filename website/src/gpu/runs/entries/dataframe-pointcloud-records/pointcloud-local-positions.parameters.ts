// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// Distance of the survey from the projection origin (the magnitude of the absolute coordinates),
// and the extent of the surveyed patch.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'originDistance',
    label: 'Data: coordinate magnitude',
    min: 0.1,
    max: 10,
    step: 0.1,
    default: 4.5,
    unit: 'Mm'
  },
  {id: 'extent', label: 'Data: patch extent', min: 5, max: 40, step: 1, default: 12, unit: 'm'}
];
