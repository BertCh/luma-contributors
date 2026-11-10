// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// windDirection sets the angle of the uniform `wind` prop; verticalExaggeration scales the relief.
export const parameters: GpuSceneParameter[] = [
  {
    id: 'windDirection',
    label: 'wind (direction)',
    min: 0,
    max: 360,
    step: 5,
    default: 20,
    unit: ' deg'
  },
  {id: 'verticalExaggeration', label: 'verticalExaggeration', min: 0, max: 30, step: 1, default: 8}
];
